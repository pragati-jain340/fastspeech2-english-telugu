import hashlib, json, math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .frontend import encode
from .plotting import variance_plots


@torch.inference_mode()
def generate(model, state, language, text, output, name='custom', spoken_text=None,
             vocoder=None, checkpoint=None, max_frames=4000):
    output=Path(output);device=next(model.parameters()).device
    spoken=text if spoken_text is None else spoken_text
    tokens,ids=encode(spoken,language,state['metadata'])
    t=torch.tensor([ids],dtype=torch.long,device=device);lengths=torch.tensor([len(ids)],device=device)
    cfg=state['model_config'];frame_level=cfg.get('frame_level',False)
    if 'frame_level' in cfg:
        pred=model(t,lengths,max_frames=max_frames)
    else:
        # Legacy forward lacks a safety-limit parameter. Check on the encoder
        # before length regulation so long input cannot allocate unbounded frames.
        from .legacy_model import padding_mask, add_positions
        pad=padding_mask(lengths,t.shape[1]);x=add_positions(model.embedding(t)).masked_fill(pad[...,None],0)
        for b in model.encoder: x=b(x,pad)
        dp=model.predictors['duration'](x,pad)
        if int(torch.expm1(dp.clamp(0,math.log1p(100))).round().clamp_min(1).sum())>max_frames:
            raise ValueError('Predicted duration exceeds safety limit')
        pred=model(t,lengths)
    frames=int(pred['mel_lengths'][0]);units=frames if frame_level else len(ids)
    mel=pred['mel'][0,:frames].float().cpu().numpy()
    raw=pred['duration'][0,:len(ids)]
    d=torch.expm1(raw.clamp(0,math.log1p(100))).round().long().clamp_min(1).cpu().numpy()
    if int(d.sum())!=frames or not np.isfinite(mel).all(): raise ValueError('Invalid predicted mel or duration sum')
    p,e,v=(pred[k][0,:units].float().cpu().numpy() for k in ('pitch','energy','voicing'))
    if not all(np.isfinite(a).all() for a in (p,e,v)): raise ValueError('Nonfinite prosody')
    meta=state['metadata']
    arrays=dict(mel=mel,tokens=np.asarray(tokens),token_ids=np.asarray(ids),duration_log_prediction=raw.cpu().numpy(),
        duration_frames=d,pitch_normalized=p,energy_normalized=e,voicing_logits=v,voiced=v>=0,
        f0_hz=p*meta['pitch_std']+meta['pitch_mean'],energy_original_units=e*meta['energy_std']+meta['energy_mean'])
    directory=output/'predictions';directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/f'{name}.npz',**arrays)
    details=dict(id=name,language=language,original_text=text,spoken_form_before_frontend=spoken,
        normalized_model_input=''.join(tokens),tokens=tokens,frontend='identity' if language=='english' else 'uroman_tel',
        frontend_mode='MMS word regex, concatenated characters',variant='original_baseline' if 'frame_level' not in cfg else 'frame_prosody' if frame_level else 'character_baseline',
        checkpoint=str(checkpoint),epoch=state['epoch'],updates=state.get('updates'),conditioning='text_only',
        mel_frames=frames,sample_rate=22050,duration_seconds=frames*256/22050,metadata=meta,
        explicit_number_time_expansion=spoken!=text,invalid_voiced_f0_units=int(np.sum((v>=0)&(arrays['f0_hz']<=0))),
        negative_energy_units=int(np.sum(arrays['energy_original_units']<0)))
    if checkpoint:
        with Path(checkpoint).open('rb') as f: details['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    if vocoder is not None:
        wave=vocoder(mel);peak=float(np.max(np.abs(wave)));gain=min(1.,.95/peak) if peak else 1.
        wav=output/'test_generated_audios'/f'{name}.wav';wav.parent.mkdir(parents=True,exist_ok=True)
        sf.write(wav,wave*gain,22050);details.update(vocoder.metadata,gain=gain)
    variance_plots(output/'variance_adaptor_plots',name,arrays,frame_level)
    (directory/f'{name}.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf8')
    return details
