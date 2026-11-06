"""Exact centered log-magnitude mel and TorchCREPE targets from Colab."""
import io
import numpy as np
import torch

ACOUSTICS = dict(sample_rate=22050, hop=256, fft=1024, n_mels=80,
                 fmax=8000, center=True, pad_mode='reflect', window='hann',
                 mel_scale='slaney', mel_norm='slaney', log_floor=1e-5)


def mel_and_energy(raw_audio, tokens, device):
    import librosa
    import soundfile as sf
    samples, rate = sf.read(io.BytesIO(raw_audio), dtype='float32')
    start = max(0, round(tokens[0]['start_s'] * rate))
    end = min(len(samples), round(tokens[-1]['end_s'] * rate))
    if samples.ndim != 1 or end <= start:
        raise ValueError('Invalid mono waveform or alignment bounds')
    samples = samples[start:end]
    a22 = librosa.resample(samples, orig_sr=rate, target_sr=22050)
    a16 = librosa.resample(samples, orig_sr=rate, target_sr=16000)
    window = torch.hann_window(1024, device=device)
    filters = torch.as_tensor(librosa.filters.mel(sr=22050, n_fft=1024,
                              n_mels=80, fmax=8000, htk=False, norm='slaney'), device=device)
    mag = torch.stft(torch.as_tensor(a22, device=device), n_fft=1024,
                     hop_length=256, window=window, center=True,
                     pad_mode='reflect', return_complex=True).abs()
    mel = (filters @ mag).clamp_min(1e-5).log().T.cpu().numpy().astype('float32')
    energy = torch.linalg.vector_norm(mag, dim=0).cpu().numpy().astype('float32')
    return mel, energy, a16, start / rate, len(a22) / 22050


@torch.inference_mode()
def extract(raw_audio, tokens, device='cpu'):
    import torchcrepe
    mel, energy, a16, crop_start, seconds = mel_and_energy(raw_audio, tokens, device)
    frames = len(mel)
    if frames < len(tokens):
        raise ValueError('Fewer frames than characters')
    midpoints = np.array([(a['end_s'] + b['start_s']) / 2 for a,b in zip(tokens,tokens[1:])])
    interior = np.rint((midpoints - crop_start) * 22050 / 256).astype('int64')
    indices = np.arange(1,len(tokens))
    slack = np.clip(interior-indices,0,frames-len(tokens))
    boundaries = np.r_[0,np.maximum.accumulate(slack)+indices,frames]
    x16 = torch.from_numpy(a16).unsqueeze(0)
    f0, periodicity = torchcrepe.predict(x16,16000,hop_length=160,fmin=65,fmax=500,
        model='tiny',decoder=torchcrepe.decode.weighted_argmax,return_periodicity=True,
        batch_size=512 if str(device).startswith('cuda') else 32,device=str(device))
    periodicity = torchcrepe.threshold.Silence(-60.0)(periodicity.cpu(),x16,16000,160)
    pick = np.clip(np.rint(np.arange(frames)*256/22050*100).astype('int64'),0,f0.shape[-1]-1)
    f0 = f0.cpu().numpy()[0,pick]
    voiced = (periodicity.numpy()[0,pick]>=.25)&np.isfinite(f0)
    pitch=np.zeros(len(tokens),'float32'); token_energy=np.zeros_like(pitch)
    for i,(a,b) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        values=f0[a:b][voiced[a:b]]
        if len(values): pitch[i]=values.mean()
        token_energy[i]=energy[a:b].mean()
    arrays=dict(tokens=np.asarray([t['character'] for t in tokens]),mel=mel,
                duration=np.diff(boundaries).astype('int32'),pitch=pitch,energy=token_energy)
    frame=dict(f0=np.where(voiced,f0,0).astype('float32'),energy=energy,voiced=voiced)
    validate_feature(arrays)
    return arrays,frame,seconds


def validate_feature(a):
    n=len(a['tokens']); mel=a['mel']; d=a['duration']
    if n<1 or mel.ndim!=2 or mel.shape[1]!=80 or len(mel)<1:
        raise ValueError('Invalid token/mel shape')
    if any(a[k].shape!=(n,) for k in ('duration','pitch','energy')):
        raise ValueError('Invalid character target shape')
    if (d<=0).any() or not np.equal(d,np.round(d)).all() or int(d.sum())!=len(mel):
        raise ValueError('Invalid duration grid')
    if any(not np.isfinite(a[k]).all() for k in ('mel','pitch','energy','duration')):
        raise ValueError('Nonfinite feature')
