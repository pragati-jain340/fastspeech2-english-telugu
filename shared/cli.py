import argparse, csv, json, random
from pathlib import Path


def main(language, action):
    parser=argparse.ArgumentParser(description=f'{language.title()} FastSpeech 2: {action}',
        epilog='Colab setup and checkpoint downloads: START_HERE.md. Use filesystem paths for --root and --checkpoint.')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/language.title()/'config.yaml')
    parser.add_argument('--root',type=Path,help='Persistent data root (Drive recommended)')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    if action=='preprocess':
        parser.add_argument('--stage',choices=['download','select','align','finalize','features','import-cache','all'],default='all')
        parser.add_argument('--targets',type=Path)
        parser.add_argument('--run',type=Path,help='Historical run required for import-cache')
    elif action=='train':
        parser.add_argument('--run',type=Path)
        parser.add_argument('--targets',type=Path)
        parser.add_argument('--epochs',type=int,help='TOTAL target epochs, never additional epochs')
        parser.add_argument('--variant',choices=['frame_prosody','character_baseline','both'],default='both')
        parser.add_argument('--resume',action='store_true',help='Require last.pt and restore optimizer/RNG/partial epoch')
    else:
        parser.add_argument('--checkpoint',type=Path)
        parser.add_argument('--variant',choices=['frame_prosody','character_baseline','original_baseline'],default='frame_prosody')
        parser.add_argument('--text',help='Original text; preserved in metadata')
        parser.add_argument('--spoken-text',help='Explicit number/date/time expansion')
        parser.add_argument('--assignment',action='store_true',help='Generate five prescribed utterances')
        parser.add_argument('--output',type=Path)
        parser.add_argument('--vocoder-cache',type=Path)
        parser.add_argument('--mel-only',action='store_true',help='Skip waveform synthesis for diagnostics')
        parser.add_argument('--max-frames',type=int,default=4000)
    args=parser.parse_args()
    # Help must work without acoustic or training dependencies and without side effects.
    import yaml
    import numpy as np
    import torch
    cfg=yaml.safe_load(args.config.read_text(encoding='utf8'))
    if cfg['language']!=language: parser.error('Configuration language mismatch')
    root=args.root or Path(cfg['data_root']);root=root.expanduser()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): parser.error('CUDA is unavailable; use --device cpu')
    if device.type=='cpu': torch.set_num_threads(min(4,torch.get_num_threads()))
    if action=='preprocess':
        from . import preprocessing as prep
        root.mkdir(parents=True,exist_ok=True)
        targets=args.targets or root/'runs'/language/'frame_targets'
        stages=['download','select','align','finalize','features'] if args.stage=='all' else [args.stage]
        for stage in stages:
            print('Preprocessing',language,stage,'on',device)
            if stage=='align': prep.align(root,language,device)
            elif stage=='features': prep.features(root,language,device,targets)
            elif stage=='import-cache':
                if not args.run: parser.error('--run required for import-cache')
                from .cache import import_cache
                import_cache(root,args.run,targets,language)
            else: getattr(prep,stage)(root,language)
    elif action=='train':
        from .training import train
        from .data import prepare
        run=args.run or root/'runs'/language
        variants=[(v,v=='frame_prosody') for v in (['frame_prosody','character_baseline'] if args.variant=='both' else [args.variant])]
        metadata_path=root/'features/mms_char_crepe_gpu_v1/training_metadata.json'
        if not metadata_path.is_file():
            parser.error(f'Missing training features: {metadata_path}. For new training, run {language.title()}/preprocess.py --root "{root}" --stage all. For historical resume, restore the exact matching feature cache. See START_HERE.md.')
        meta=json.loads(metadata_path.read_text())[language]
        epochs=args.epochs if args.epochs is not None else cfg['training']['max_epochs']
        if epochs<1: parser.error('Total target epochs must be positive')
        for v,_ in variants:
            last=run/v/'last.pt'
            if args.resume and not last.is_file(): parser.error(f'Missing resume checkpoint: {last}')
            if last.exists() and not args.resume: parser.error(f'{last} exists; pass --resume or choose a new --run')
            if last.exists():
                from .checkpoints import load_checkpoint
                _,saved=load_checkpoint(last,device,language,v)
                old=saved['run_config']
                for key in ('seed','batch_size','sample_rate','hop','fft','center','pad_mode','window','n_mels','fmax'):
                    expected=cfg['training'].get(key,cfg['acoustics'].get(key))
                    if old.get(key)!=expected: parser.error(f'Resume configuration differs for {key}')
                if saved['metadata']!=meta: parser.error('Resume vocabulary/normalization differs')
                if epochs<int(saved['epoch']): parser.error('Target epochs cannot be earlier than the saved epoch')
        settings=dict(cfg['training'],max_epochs=epochs,**cfg['acoustics'],models=[v for v,_ in variants],language=language)
        random.seed(settings['seed']);np.random.seed(settings['seed']);torch.manual_seed(settings['seed'])
        if device.type=='cuda': torch.cuda.manual_seed_all(settings['seed'])
        items=prepare(root/'features/mms_char_crepe_gpu_v1',args.targets or run/'frame_targets',language,meta,run)
        run.mkdir(parents=True,exist_ok=True);(run/'run_config.json').write_text(json.dumps(settings,indent=2))
        train(items,meta,run,settings,device,variants)
    else:
        if not args.text and not args.assignment: parser.error('Pass --text or --assignment')
        if args.text and args.assignment: parser.error('Choose --text or --assignment')
        from .checkpoints import load_checkpoint
        from .inference import generate
        checkpoint=args.checkpoint or root/'runs'/language/args.variant/'best_mel.pt'
        if not checkpoint.is_file():
            parser.error(f'Missing checkpoint: {checkpoint}. Download and extract the linked backup, then run scripts/import_assets.py --source BACKUP_FOLDER --destination "{root}". See START_HERE.md for download links and folder layout.')
        model,state=load_checkpoint(checkpoint,device,language,args.variant)
        output=args.output or root/'Results'/language.title()
        vocoder=None
        if not args.mel_only:
            from .vocoder import Vocoder
            vocoder=Vocoder(args.vocoder_cache or root/'cache/hifigan',device)
        if args.assignment:
            tests=json.loads((Path(__file__).resolve().parents[1]/'test_sentences.json').read_text(encoding='utf8'))[language.title()]
        else: tests={'custom':dict(original_text=args.text,spoken_text=args.spoken_text)}
        inventory=[]
        for name,r in tests.items():
            try:
                details=generate(model,state,language,r['original_text'],output,name,r.get('spoken_text'),vocoder,checkpoint,args.max_frames)
                inventory.append(dict(language=language,id=name,status='mel_only' if args.mel_only else 'generated',epoch=state['epoch'],seconds=details['duration_seconds'],error=''))
            except Exception as e:
                inventory.append(dict(language=language,id=name,status='FAILED',epoch=state['epoch'],seconds='',error=f'{type(e).__name__}: {e}'))
        output.mkdir(parents=True,exist_ok=True)
        with (output/'generation_inventory.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
        print(json.dumps(inventory,ensure_ascii=False,indent=2))
        if any(r['status']=='FAILED' for r in inventory): raise SystemExit(1)
