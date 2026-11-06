"""Copy selected checkpoint runs from a local/Drive backup, without raw data."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Extracted backup folder path (not a Drive sharing URL)')
    p.add_argument('--destination',type=Path,required=True,help='Persistent root; creates runs/english and runs/telugu')
    args=p.parse_args()
    if not args.source.is_dir():
        p.error(f'Backup folder not found: {args.source}. Download and extract the English/Telugu backups first; pass the extracted parent folder path. See START_HERE.md for links and layout.')
    from shared.checkpoints import load_checkpoint
    log=[]
    for language in ['english','telugu']:
        candidates=list(args.source.rglob(f'{language}_full_frame_prosody_v1'))
        # A source may be the run directory itself.
        if args.source.name==f'{language}_full_frame_prosody_v1': candidates.insert(0,args.source)
        valid=[r for r in candidates if all((r/v/'last.pt').is_file() and (r/v/'best_mel.pt').is_file() for v in ['frame_prosody','character_baseline'])]
        if not valid: print(f'{language}: no complete backup found');continue
        def rank(r):
            import torch
            return torch.load(r/'frame_prosody/last.pt',map_location='cpu',weights_only=True)['updates']
        source=max(valid,key=rank);dest=args.destination/'runs'/language
        for variant in ['frame_prosody','character_baseline']:
            for name in ['last.pt','best_mel.pt','metrics.csv']:
                src=source/variant/name;dst=dest/variant/name;dst.parent.mkdir(parents=True,exist_ok=True)
                if name.endswith('.pt'): load_checkpoint(src,'cpu',language,variant)
                with src.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
                if dst.exists():
                    with dst.open('rb') as f: existing=hashlib.file_digest(f,'sha256').hexdigest()
                    if digest!=existing: raise ValueError(f'Refusing to overwrite different asset: {dst}')
                else: shutil.copy2(src,dst)
                log.append(dict(source=str(src),destination=str(dst),sha256=digest))
        for name in ['train_sources.json','validation_sources.json','run_config.json','summary.json']:
            src=source/name;dst=dest/name
            if not src.exists(): continue
            if dst.exists() and src.read_bytes()!=dst.read_bytes(): raise ValueError(f'Different metadata at {dst}')
            if not dst.exists(): shutil.copy2(src,dst)
    if not log:
        p.error('No complete checkpoint runs found. The extracted backup must contain english_full_frame_prosody_v1 or telugu_full_frame_prosody_v1, with both model variants and their best_mel.pt / last.pt files. See START_HERE.md.')
    args.destination.mkdir(parents=True,exist_ok=True)
    (args.destination/'import_manifest.json').write_text(json.dumps(log,indent=2))
    print('Imported',len(log),'verified files')


if __name__=='__main__': main()
