import hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import torch
from .features import validate_feature, ACOUSTICS


def audio_column(parquet):
    """Discover real embedded audio; English uses audio_filepath, Telugu audio."""
    import pyarrow as pa
    candidates=[]
    for field in parquet.schema_arrow:
        t=field.type
        if (pa.types.is_struct(t) and 'bytes' in [c.name for c in t]) or pa.types.is_binary(t) or pa.types.is_large_binary(t):
            candidates.append(field.name)
    preferred=[n for n in candidates if n.lower() in ('audio','wav','speech','audio_filepath')]
    if len(preferred)==1: return preferred[0]
    if len(candidates)==1: return candidates[0]
    raise ValueError(f'Ambiguous embedded audio schema: {candidates}')


def selected_audio(path, rows):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path); col=audio_column(pf); offset=0;found=set()
    for batch in pf.iter_batches(columns=[col],batch_size=16):
        for j,audio in enumerate(batch.column(0).to_pylist()):
            idx=offset+j
            if idx in rows:
                raw=audio.get('bytes') if isinstance(audio,dict) else audio
                if not raw: raise ValueError(f'No embedded audio: {path}:{idx}')
                found.add(idx); yield idx,raw,rows[idx]
        offset+=batch.num_rows
        if len(found)==len(rows): break
    if len(found)!=len(rows): raise ValueError(f'Missing source rows in {path}')


def records_for(features, language):
    records={'train':[],'validation':[]};seen=set()
    paths=sorted(list(Path(features).glob(f'{language}_*.zip'))+list(Path(features).glob(f'{language}_*.fs2zip')))
    for p in paths:
        with zipfile.ZipFile(p) as z:
            for r in json.loads(z.read('index.json')):
                if r.get('status')!='ok' or r.get('split') not in records: continue
                key=(r['shard'].replace('\\','/'),int(r['row_index']))
                if key in seen: raise ValueError(f'Duplicate or overlapping source: {key}')
                seen.add(key)
                if r['member'] not in z.namelist(): raise ValueError('Feature member missing')
                records[r['split']].append(dict(r,archive=str(p)))
    if not all(records.values()): raise FileNotFoundError(f'No complete feature archives in {features}. Run preprocessing first.')
    return records


def identity(records):
    return {s:[{k:(Path(r[k]).name if k=='archive' else r[k]) for k in ('archive','member','shard','row_index','split','text')} for r in rs] for s,rs in records.items()}


def fingerprint(records, metadata):
    payload=dict(sources=identity(records),metadata=metadata,acoustics=ACOUSTICS)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def read_feature(r):
    with zipfile.ZipFile(r['archive']) as z, np.load(io.BytesIO(z.read(r['member'])),allow_pickle=False) as a:
        data={k:a[k].copy() for k in ('tokens','mel','duration','pitch','energy')}
    validate_feature(data)
    return data


def prepare(features, targets, language, metadata, output):
    records=records_for(features,language); output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for split,rs in records.items():
        p=output/f'{split}_sources.json'
        if p.exists():
            old={split:json.loads(p.read_text(encoding='utf8'))}
            if identity(old)[split]!=identity({split:rs})[split]:
                raise ValueError('Loader ordering/source manifest changed; use a new run directory')
        p.write_text(json.dumps([dict(loader_index=i,**r) for i,r in enumerate(rs)],ensure_ascii=False,indent=2),encoding='utf8')
    signature=fingerprint(records,metadata)
    stamp=Path(targets)/f'{language}_cache.json'
    if not stamp.exists():
        raise ValueError('Frame cache lacks provenance. Run preprocess --stage features to recreate targets, or --stage import-cache to validate historical cache.')
    saved=json.loads(stamp.read_text(encoding='utf8'))
    if saved['fingerprint']!=signature: raise ValueError('Incompatible feature/frame cache fingerprint')
    frame={'train':{},'validation':{}}
    for p in sorted(Path(targets).glob('targets_*.npz')):
        with np.load(p,allow_pickle=False) as z:
            offsets=np.r_[0,np.cumsum(z['lengths'])]
            if any(len(z[k])!=offsets[-1] for k in ('f0','energy','voiced')): raise ValueError('Invalid frame chunk')
            for j,(s,i) in enumerate(zip(z['splits'],z['indices'])):
                split=('train','validation')[int(s)];idx=int(i);a,b=offsets[j:j+2]
                if idx in frame[split]: raise ValueError('Duplicate frame index')
                frame[split][idx]=tuple(z[k][a:b].copy() for k in ('f0','energy','voiced'))
    items={s:[] for s in records}
    for s,rs in records.items():
        if set(frame[s])!=set(range(len(rs))): raise ValueError('Frame targets incomplete')
        for i,r in enumerate(rs):
            a=read_feature(r);f0,e,v=frame[s][i];n=len(a['mel'])
            if any(len(x)!=n for x in (f0,e,v)) or any(not np.isfinite(x).all() for x in (f0,e)):
                raise ValueError('Invalid frame targets')
            missing=set(a['tokens'])-set(metadata['vocabulary'])
            if missing: raise ValueError(f'Vocabulary mismatch: {missing}')
            ids=[metadata['vocabulary'][str(t)] for t in a['tokens']]
            voiced=a['pitch']>0;p=np.zeros_like(a['pitch'],dtype='float32');p[voiced]=(a['pitch'][voiced]-metadata['pitch_mean'])/metadata['pitch_std']
            fp=np.zeros(n,'float32');fp[v]=(f0[v]-metadata['pitch_mean'])/metadata['pitch_std']
            d=dict(tokens=torch.tensor(ids),mel=torch.from_numpy(a['mel']).float(),duration=torch.from_numpy(a['duration']).long(),
                   pitch=torch.from_numpy(p),energy=torch.from_numpy((a['energy']-metadata['energy_mean'])/metadata['energy_std']).float(),voiced=torch.from_numpy(voiced),
                   frame_pitch=torch.from_numpy(fp),frame_energy=torch.from_numpy((e-metadata['energy_mean'])/metadata['energy_std']).float(),frame_voiced=torch.from_numpy(v))
            items[s].append(d)
    return items
