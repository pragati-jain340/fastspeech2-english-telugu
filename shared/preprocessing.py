"""Independent Colab preprocessing with explicit stages and provenance checks."""
import hashlib, io, json, random, re, unicodedata, zipfile
from pathlib import Path
import numpy as np
import torch
from .data import audio_column, selected_audio, records_for, read_feature, fingerprint
from .features import extract, ACOUSTICS

REPOS={'english':'ai4bharat/Svarah','telugu':'SPRINGLab/IndicTTS_Telugu'}


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def write_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8');tmp.replace(path)


def download(root, language):
    from huggingface_hub import HfApi, snapshot_download
    p=root/'dataset_revisions.json'
    revisions=json.loads(p.read_text()) if p.exists() else {}
    if language not in revisions:
        repo=REPOS[language]
        revisions[language]=dict(repo_id=repo,revision=HfApi().dataset_info(repo).sha)
        p.write_text(json.dumps(revisions,indent=2))
    r=revisions[language]
    snapshot_download(repo_id=r['repo_id'],repo_type='dataset',revision=r['revision'],local_dir=str(root/'datasets'/language.title()))


def select(root, language):
    import pyarrow.parquet as pq
    import soundfile as sf
    shards=sorted((root/'datasets'/language.title()).rglob('*.parquet'))
    if not shards: raise FileNotFoundError('Download the source Parquet shards first')
    candidates=[];seen_text=set();seen_audio=set()
    for shard in shards:
        pf=pq.ParquetFile(shard);col=audio_column(pf);row_index=0
        for batch in pf.iter_batches(columns=[col,'text','gender'],batch_size=8):
            for row in batch.to_pylist():
                idx=row_index;row_index+=1
                if language=='telugu' and row['gender']!=0: continue
                text=' '.join((row['text'] or '').split())
                key=''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())
                a=row[col];raw=a.get('bytes') if isinstance(a,dict) else a
                if not key or key in seen_text or not raw: continue
                info=sf.info(io.BytesIO(raw))
                if info.channels!=1 or not 2<=info.duration<=20: continue
                digest=hashlib.sha256(raw).hexdigest()
                if digest in seen_audio: continue
                seen_text.add(key);seen_audio.add(digest)
                candidates.append(dict(shard=shard.relative_to(root/'datasets').as_posix(),row_index=idx,
                    source_audio=a.get('path') if isinstance(a,dict) else None,text=text,duration_s=info.duration,
                    sample_rate=info.samplerate,has_numerals=any(c.isdigit() for c in text)))
    random.Random(42).shuffle(candidates)
    splits={'train':[],'validation':[],'reserve':[]};totals=dict.fromkeys(splits,0.)
    for r in candidates:
        s='validation' if totals['validation']<3600 else 'train' if totals['train']<18000 else 'reserve'
        splits[s].append(r);totals[s]+=r['duration_s']
    for s,rs in splits.items():
        write_jsonl(root/'manifests'/f'{language}_{s}.jsonl',rs)
        print(language,s,len(rs),totals[s]/3600,'hours before alignment')


def align(root, language, device):
    import soundfile as sf
    from torchaudio.pipelines import MMS_FA
    from uroman import Uroman
    net=MMS_FA.get_model().to(device).eval();tokenizer=MMS_FA.get_tokenizer();aligner=MMS_FA.get_aligner();romanizer=Uroman()
    for group,splits in [('mms_char_v1',('train','validation')),('mms_char_reserve_v1',('reserve',))]:
        groups={}
        for split in splits:
            for r in jsonl(root/'manifests'/f'{language}_{split}.jsonl'):
                groups.setdefault(r['shard'],{})[int(r['row_index'])]=dict(r,split=split)
        for shard,rows in sorted(groups.items()):
            dest=root/'alignments'/group/f'{language}_{Path(shard).stem}.jsonl'
            signature=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
            stamp=dest.with_suffix('.provenance.json')
            if dest.exists() and stamp.exists() and json.loads(stamp.read_text())['source_hash']==signature:
                print('Reuse',dest);continue
            results=[]
            for idx,raw,r in selected_audio(root/'datasets'/shard,rows):
                result=dict(r)
                try:
                    if r['has_numerals']: raise ValueError('numeral_needs_normalization')
                    text=romanizer.romanize_string(r['text'],lcode='tel') if language=='telugu' else r['text']
                    words=re.findall(r"[a-z]+(?:'[a-z]+)*",text.lower())
                    if not words: raise ValueError('no_alignable_tokens')
                    samples,rate=sf.read(io.BytesIO(raw),dtype='float32')
                    if rate!=MMS_FA.sample_rate or samples.ndim!=1: raise ValueError('expected_16k_mono')
                    with torch.inference_mode():
                        emission,_=net(torch.from_numpy(samples)[None].to(device));spans=aligner(emission[0],tokenizer(words))
                    scale=len(samples)/rate/emission.shape[1];tokens=[]
                    for word,ws in zip(words,spans,strict=True):
                        for char,span in zip(word,ws,strict=True):
                            tokens.append(dict(character=char,start_s=round(span.start*scale,4),end_s=round(span.end*scale,4),score=float(span.score)))
                    if not tokens: raise ValueError('empty_alignment')
                    result.update(status='aligned',tokens=tokens)
                except torch.cuda.OutOfMemoryError: raise
                except (ValueError,RuntimeError,KeyError,IndexError) as e: result.update(status='skipped',reason=str(e))
                results.append(result)
            write_jsonl(dest,results);stamp.write_text(json.dumps(dict(source_hash=signature)))


def finalize(root, language):
    groups=[]
    for d in ['mms_char_v1','mms_char_reserve_v1']:
        rows=[]
        for p in sorted((root/'alignments'/d).glob(f'{language}_*.jsonl')):
            for r in jsonl(p):
                if r['status']=='aligned':
                    seconds=r['tokens'][-1]['end_s']-r['tokens'][0]['start_s']
                    if seconds<=0: raise ValueError('Invalid alignment duration')
                    rows.append(dict(r,aligned_duration_s=seconds))
        if not rows: raise ValueError(f'Missing {d} alignments')
        groups.append(rows)
    original,reserve=groups
    ids=[(r['shard'],r['row_index']) for r in original+reserve]
    if len(set(ids))!=len(ids): raise ValueError('Alignment source overlap')
    random.Random(42).shuffle(reserve);offset=0
    for s,target in [('train',18000),('validation',3600)]:
        rows=[r for r in original if r['split']==s];seconds=sum(r['aligned_duration_s'] for r in rows)
        while seconds<target and offset<len(reserve):
            r=dict(reserve[offset],split=s);rows.append(r);seconds+=r['aligned_duration_s'];offset+=1
        if seconds<target: raise ValueError(f'{language} {s} has insufficient aligned duration')
        write_jsonl(root/'manifests/aligned_final_v1'/f'{language}_{s}.jsonl',rows)


def metadata_for(records):
    symbols=set();pitch=[];energy=[]
    for r in records['train']:
        a=read_feature(r);symbols.update(a['tokens'].tolist());pitch.append(a['pitch'][a['pitch']>0]);energy.append(a['energy'])
    p=np.concatenate(pitch).astype('float64');e=np.concatenate(energy).astype('float64')
    if not len(p): raise ValueError('No voiced training characters')
    return dict(vocabulary={t:i for i,t in enumerate(sorted(symbols),2)},pad_id=0,unk_id=1,
        pitch_mean=float(p.mean()),pitch_std=max(float(p.std()),1e-6),energy_mean=float(e.mean()),energy_std=max(float(e.std()),1e-6))


def features(root, language, device, targets):
    dest=root/'features/mms_char_crepe_gpu_v1';dest.mkdir(parents=True,exist_ok=True)
    targets=Path(targets);targets.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split in ('train','validation'): rows+=jsonl(root/'manifests/aligned_final_v1'/f'{language}_{split}.jsonl')
    ids=[(r['shard'],r['row_index']) for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Source-record overlap')
    groups={}
    for r in rows: groups.setdefault(r['shard'],{})[int(r['row_index'])]=r
    for shard,selected in sorted(groups.items()):
        path=dest/f'{language}_{Path(shard).stem}.zip';stamp=path.with_suffix('.provenance.json')
        sig=hashlib.sha256(json.dumps(dict(rows=selected,acoustics=ACOUSTICS),sort_keys=True).encode()).hexdigest()
        if path.exists() and stamp.exists() and json.loads(stamp.read_text())['fingerprint']==sig:
            print('Reuse',path);continue
        temp=path.with_suffix('.partial');index=[]
        with zipfile.ZipFile(temp,'w') as z:
            for idx,raw,r in selected_audio(root/'datasets'/shard,selected):
                a,frame,seconds=extract(raw,r['tokens'],device)
                member=f"{r['split']}/{idx:06d}.npz";buf=io.BytesIO();np.savez(buf,**a);z.writestr(member,buf.getvalue())
                # Frame data is colocated to allow safe reconstruction in exact archive order.
                buf=io.BytesIO();np.savez(buf,**frame);z.writestr('frame/'+member,buf.getvalue())
                index.append(dict(shard=shard,row_index=idx,split=r['split'],text=r['text'],member=member,status='ok',seconds=seconds))
            z.writestr('index.json',json.dumps(index,ensure_ascii=False))
        temp.replace(path);stamp.write_text(json.dumps(dict(fingerprint=sig)))
    records=records_for(dest,language);meta=metadata_for(records)
    p=dest/'training_metadata.json';all_meta=json.loads(p.read_text()) if p.exists() else {};all_meta[language]=meta;p.write_text(json.dumps(all_meta,indent=2))
    # Avoid stale surplus chunks when a different source selection is used.
    if list(targets.glob('targets_*.npz')):
        stamp=targets/f'{language}_cache.json'
        if not stamp.exists() or json.loads(stamp.read_text())['fingerprint']!=fingerprint(records,meta):
            raise ValueError('Use an empty target directory for a changed source manifest')
    chunk=[];number=0
    for split,rs in records.items():
        for i,r in enumerate(rs):
            with zipfile.ZipFile(r['archive']) as z,np.load(io.BytesIO(z.read('frame/'+r['member'])),allow_pickle=False) as a:
                chunk.append((split,i,{k:a[k].copy() for k in ('f0','energy','voiced')}))
            if len(chunk)==32:
                save_chunk(targets,number,chunk);number+=1;chunk=[]
    if chunk: save_chunk(targets,number,chunk)
    (targets/f'{language}_cache.json').write_text(json.dumps(dict(fingerprint=fingerprint(records,meta)),indent=2))


def save_chunk(targets,number,chunk):
    p=targets/f'targets_{number:04d}.npz';tmp=p.with_suffix('.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f,splits=np.array([0 if s=='train' else 1 for s,i,a in chunk],dtype='int8'),
            indices=np.array([i for s,i,a in chunk],dtype='int32'),lengths=np.array([len(a['f0']) for s,i,a in chunk],dtype='int32'),
            **{k:np.concatenate([a[k] for s,i,a in chunk]) for k in ('f0','energy','voiced')})
    tmp.replace(p)
