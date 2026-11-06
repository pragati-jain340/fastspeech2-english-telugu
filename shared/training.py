"""Original full-run training, masks, AdamW, deterministic shuffles and resume."""
import csv, json, math, os
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .model import PilotFS2, padmask
from .losses import loss_for
from .checkpoints import save_checkpoint

def train(items, meta, output, cfg, device, variants, vocode=None):
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    SEED, BATCH, EPOCHS = cfg['seed'], cfg['batch_size'], cfg['max_epochs']
    SAMPLE_EVERY, STEP_SAVE_EVERY, SR = cfg['sample_every'], 100, 22050
    TEST_IDS = list(range(min(3,len(items['train']))))
    def collate(examples):
        b = {k: pad_sequence([e[k] for e in examples], batch_first=True) for k in examples[0]}
        b['token_lengths'] = torch.tensor([len(e['tokens']) for e in examples])
        b['mel_lengths'] = torch.tensor([len(e['mel']) for e in examples])
        if not torch.equal(b['duration'].sum(1), b['mel_lengths']):
            raise RuntimeError('Duration sum mismatch')
        return {k: v.to(device) for k, v in b.items()}

    @torch.inference_mode()
    def validate(model, frame_level):
        model.eval()
        mel_sum = 0.0
        nframes = 0
        total_sum = 0.0
        nclips = 0
        for start in range(0, len(items['validation']), BATCH):
            b = collate(items['validation'][start:start + BATCH])
            out = model(b['tokens'], b['token_lengths'], targets=b)
            total, mel = loss_for(out, b, frame_level)
            n = int(b['mel_lengths'].sum())
            count = len(b['tokens'])
            mel_sum += mel.item() * n
            nframes += n
            total_sum += total.item() * count
            nclips += count
        return (mel_sum / nframes, total_sum / nclips)

    @torch.inference_mode()
    def export_samples(model, label, epoch):
        model.eval()
        folder = OUT / label / 'samples' / f'epoch_{epoch:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        for split, chosen in (('validation', range(min(3,len(items['validation'])))), ('train', TEST_IDS)):
            for idx in chosen:
                b = collate([items[split][idx]])
                n = int(b['mel_lengths'][0])
                ref = b['mel'][0, :n].cpu().numpy()
                fixed = model(b['tokens'], b['token_lengths'], targets=b)['mel'][0, :n].cpu().numpy()
                result = dict(reference=ref, recorded_prosody=fixed, reference_conditioned_mae=np.array(np.mean(np.abs(ref - fixed)), dtype='float32'))
                issue = None
                try:
                    free = model(b['tokens'], b['token_lengths'], max_frames=max(1500, 3 * n))['mel'][0].cpu().numpy()
                    result['text_only'] = free
                except ValueError as e:
                    issue = str(e)
                    print(f'{label} {split} {idx}: text-only skipped: {issue}')
                np.savez_compressed(folder / f'{split}_{idx}.npz', **result)
                if vocode is not None:
                    try:
                        waves = {key: vocode(mel) for key, mel in result.items() if key in ('reference', 'recorded_prosody', 'text_only')}
                        peak = max((float(np.max(np.abs(w))) for w in waves.values()))
                        gain = min(1.0, 0.95 / peak) if peak else 1.0
                        for name, w in waves.items():
                            sf.write(folder / f'{split}_{idx}_{name}.wav', w * gain, SR)
                    except Exception as e:
                        print(f'Audio export failed for {label} {split} {idx}: {e}; mel NPZ remains saved')
        print(f'Saved sample mels/audio: {folder}', flush=True)

    fields=['model','epoch','updates','train_mel_mae','train_total','val_mel_mae','val_total','best_val_mel_mae','learning_rate']
    def write_metrics(path,rows):
        temp=path.with_suffix('.partial')
        with temp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        os.replace(temp,path)

    def save_checkpoint(path,state):
        temp=path.with_suffix('.partial')
        torch.save(state,temp)
        os.replace(temp,path)

    for label,frame_level in variants:
        folder=OUT/label;folder.mkdir(exist_ok=True)
        last=folder/'last.pt';best_path=folder/'best_mel.pt'
        complete=folder/'completed.json'
        if complete.exists():
            completed_epoch=int(json.loads(complete.read_text())['last_epoch'])
            actual_epoch=int(torch.load(last,map_location='cpu',weights_only=True)['epoch']) if last.is_file() else -1
            if completed_epoch>=EPOCHS and actual_epoch==completed_epoch and last.is_file() and best_path.is_file():
                print(f'{label}: completed through epoch {completed_epoch}; reusing saved checkpoints and metrics')
                continue
            print(f'{label}: extending saved run from epoch {completed_epoch} to {EPOCHS}')
        metrics_path=folder/'metrics.csv'
        history=list(csv.DictReader(metrics_path.open())) if metrics_path.exists() else []
        torch.manual_seed(SEED)
        if device.type=='cuda': torch.cuda.manual_seed_all(SEED)
        config=dict(vocab_size=len(meta['vocabulary'])+2,frame_level=frame_level,width=128)
        model=PilotFS2(**config).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        start_epoch=1;updates=0;best=float('inf');best_epoch=0
        resume_offset=0;resume_stats=None
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=True)
            if saved['model_config']!=config or saved['batch_size']!=BATCH or saved['metadata']!=meta:
                raise RuntimeError(f'Incompatible checkpoint {last}')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for k,v in state.items():
                    if isinstance(v,torch.Tensor): state[k]=v.to(device)
            resume_offset=int(saved.get('next_batch_start',len(items['train'])))
            if not 0<=resume_offset<=len(items['train']):
                raise RuntimeError(f'Invalid checkpoint offset {resume_offset}')
            partial_epoch=resume_offset<len(items['train'])
            start_epoch=int(saved['epoch']) if partial_epoch else int(saved['epoch'])+1
            if partial_epoch:
                resume_stats=saved['train_accumulators']
                if resume_offset<=0 or resume_offset%BATCH:
                    raise RuntimeError(f'Invalid partial-epoch checkpoint offset {resume_offset}')
            updates=int(saved['updates'])
            best=float(saved['best_val_mel_mae']);best_epoch=int(saved['best_epoch'])
            torch.set_rng_state(saved['torch_rng'].cpu())
            if device.type=='cuda' and saved['cuda_rng']:
                torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
            latest_row=saved.get('latest_row')
            if (len(history)==start_epoch-2 and latest_row is not None
                    and int(latest_row['epoch'])==start_epoch-1):
                history.append(saved['latest_row'])
                write_metrics(metrics_path,history)
            if len(history)!=start_epoch-1:
                raise RuntimeError(f'{label}: metrics and checkpoint disagree; inspect before resuming')
            if int(saved['epoch'])==best_epoch:
                if not best_path.exists():
                    save_checkpoint(best_path,saved)
                else:
                    recorded_best=torch.load(best_path,map_location='cpu',weights_only=True)
                    if int(recorded_best['epoch'])!=best_epoch:
                        save_checkpoint(best_path,saved)
            print(f'Resuming {label} at epoch {start_epoch}, batch offset {resume_offset if partial_epoch else 0}, update {updates}')
        else:
            if history: raise RuntimeError(f'{label}: metrics exist but checkpoint missing')
            print(f'Starting {label} from scratch ({sum(p.numel() for p in model.parameters()):,} parameters)')
        for epoch in range(start_epoch,EPOCHS+1):
            order=np.random.default_rng(SEED+epoch).permutation(len(items['train']))
            model.train()
            if epoch==start_epoch and resume_stats is not None:
                mel_sum,total_sum,nframes,nclips=resume_stats
                first_batch=resume_offset
            else:
                mel_sum=total_sum=0.;nframes=nclips=0
                first_batch=0
            for start in range(first_batch,len(order),BATCH):
                b=collate([items['train'][int(i)] for i in order[start:start+BATCH]])
                optimizer.zero_grad(set_to_none=True)
                out=model(b['tokens'],b['token_lengths'],targets=b)
                total,mel=loss_for(out,b,frame_level)
                if not torch.isfinite(total):raise RuntimeError(f'Nonfinite loss {label} epoch {epoch} update {updates}')
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1
                n=int(b['mel_lengths'].sum());count=len(b['tokens'])
                mel_sum+=mel.item()*n;nframes+=n;total_sum+=total.item()*count;nclips+=count
                if updates%STEP_SAVE_EVERY==0 and start+BATCH<len(order):
                    partial_payload=dict(language=cfg.get('language'),model=model.state_dict(),optimizer=optimizer.state_dict(),
                        model_config=config,metadata=meta,batch_size=BATCH,epoch=epoch,updates=updates,
                        best_val_mel_mae=best,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                        run_config=cfg,latest_row=history[-1] if history else None,
                        next_batch_start=start+BATCH,
                        train_accumulators=(mel_sum,total_sum,nframes,nclips))
                    save_checkpoint(last,partial_payload)
                    print(f'{label} epoch {epoch:02d}: saved update {updates} ({start+BATCH}/{len(order)} clips)',flush=True)
            val_mel,val_total=validate(model,frame_level)
            if not math.isfinite(val_mel) or not math.isfinite(val_total):
                raise RuntimeError(f'Nonfinite validation loss {label} epoch {epoch}')
            train_mel=mel_sum/nframes
            improved=val_mel<best
            if improved:best=val_mel;best_epoch=epoch
            row=dict(model=label,epoch=epoch,updates=updates,train_mel_mae=round(train_mel,6),
                train_total=round(total_sum/nclips,6),val_mel_mae=round(val_mel,6),
                val_total=round(val_total,6),best_val_mel_mae=round(best,6),learning_rate=optimizer.param_groups[0]['lr'])
            payload=dict(language=cfg.get('language'),model=model.state_dict(),optimizer=optimizer.state_dict(),
                model_config=config,metadata=meta,batch_size=BATCH,epoch=epoch,updates=updates,
                best_val_mel_mae=best,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                run_config=cfg,latest_row=row,next_batch_start=len(order))
            # Save complete state before CSV; the latest row repairs an interrupted write.
            save_checkpoint(last,payload)
            if improved:save_checkpoint(best_path,payload)
            history.append(row);write_metrics(metrics_path,history)
            print(f'{label} epoch {epoch:02d} | updates {updates:5d} | train mel {train_mel:.4f} | '
                  f'val mel {val_mel:.4f} | best {best:.4f} (epoch {best_epoch})',flush=True)
            if improved or epoch%SAMPLE_EVERY==0:
                export_samples(model,label,epoch)
        complete.write_text(json.dumps(dict(last_epoch=EPOCHS,last_update=updates,
            best_epoch=best_epoch,reason='matched_max_epochs'),indent=2))
        del model,optimizer
        if device.type=='cuda': torch.cuda.empty_cache()
