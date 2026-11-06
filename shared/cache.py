"""Explicit adoption of historical caches, requiring saved source ordering."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import records_for, identity, read_feature, fingerprint, selected_audio
from .features import mel_and_energy
from .preprocessing import jsonl


def import_cache(root, run, targets, language):
    root,run,targets=map(Path,(root,run,targets))
    saved=torch.load(run/'frame_prosody/last.pt',map_location='cpu',weights_only=True)
    meta=saved['metadata'];records=records_for(root/'features/mms_char_crepe_gpu_v1',language)
    original={s:json.loads((run/f'{s}_sources.json').read_text(encoding='utf8')) for s in records}
    if identity(original)!=identity(records): raise ValueError('Historical source ordering differs from feature archive ordering')
    manifests={}
    for s in records:
        for r in jsonl(root/'manifests/aligned_final_v1'/f'{language}_{s}.jsonl'):
            manifests[(r['shard'],int(r['row_index']))]=r
    frame={}
    for p in sorted(targets.glob('targets_*.npz')):
        with np.load(p,allow_pickle=False) as a:
            offsets=np.r_[0,np.cumsum(a['lengths'])]
            for j,(s,i) in enumerate(zip(a['splits'],a['indices'])):
                key=(('train','validation')[int(s)],int(i));n0,n1=offsets[j:j+2]
                if key in frame: raise ValueError('Duplicate frame targets')
                frame[key]={k:a[k][n0:n1].copy() for k in ('f0','energy','voiced')}
    expected={(s,i) for s,rs in records.items() for i in range(len(rs))}
    if set(frame)!=expected: raise ValueError('Incomplete frame cache')
    groups={}
    for s,rs in records.items():
        for i,r in enumerate(rs): groups.setdefault(r['shard'],{})[int(r['row_index'])]=(s,i,r)
    # Full raw-source mel/energy verification is intentionally expensive. Length
    # equality alone does not establish that a frame cache belongs to a source.
    for shard,rows in groups.items():
        for row,raw,(s,i,r) in selected_audio(root/'datasets'/shard,rows):
            clip=manifests[(shard,row)];a=read_feature(r)
            mel,energy,_,_,_=mel_and_energy(raw,clip['tokens'],'cpu')
            f=frame[(s,i)]
            if mel.shape!=a['mel'].shape or not np.allclose(mel,a['mel'],atol=.02,rtol=.001): raise ValueError('Raw source does not reproduce mel')
            if f['energy'].shape!=energy.shape or not np.allclose(f['energy'],energy,atol=.02,rtol=.001): raise ValueError('Frame source energy mismatch')
            if any(len(f[k])!=len(mel) for k in f) or not np.isfinite(f['f0']).all(): raise ValueError('Invalid cached F0')
            if not np.array_equal(a['tokens'],[t['character'] for t in clip['tokens']]): raise ValueError('Alignment/frontend mismatch')
    (targets/f'{language}_cache.json').write_text(json.dumps(dict(fingerprint=fingerprint(records,meta),
        verification='Every source mel and frame energy reproduced; F0 provenance inherited from matching historical source manifest, not re-estimated.'),indent=2))
