"""The same independently pretrained universal HiFi-GAN used by saved outputs."""
import importlib.util, json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

COMMIT='4769534d45265d52a904b850da5a622601885777'
REPO='AlexAlexBabarika/hifigan-universal-v1'


class Vocoder:
    def __init__(self, cache, device='cpu'):
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        code=cache/'hifi-gan'
        if not (code/'.git').is_dir():
            subprocess.run(['git','clone','https://github.com/jik876/hifi-gan.git',str(code)],check=True)
        subprocess.run(['git','-C',str(code),'checkout','--detach',COMMIT],check=True,capture_output=True)
        for name in ['g_02500000','config.json']:
            if not (cache/name).is_file():
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=REPO,filename=name,local_dir=str(cache))
        cfg=json.loads((cache/'config.json').read_text())
        if tuple(cfg[k] for k in ('sampling_rate','num_mels','n_fft','hop_size','win_size','fmax'))!=(22050,80,1024,256,1024,8000):
            raise ValueError('Vocoder configuration does not match the acoustic model')
        if int(np.prod(cfg['upsample_rates']))!=256: raise ValueError('Vocoder hop mismatch')
        # Upstream models imports its sibling utils module.
        sys.path.insert(0,str(code))
        previous=sys.modules.pop('utils',None)
        try:
            spec=importlib.util.spec_from_file_location('assignment_hifigan_models',code/'models.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(code));sys.modules.pop('utils',None)
            if previous is not None: sys.modules['utils']=previous
        self.device=torch.device(device)
        self.net=module.Generator(SimpleNamespace(**cfg)).to(self.device)
        self.net.load_state_dict(torch.load(cache/'g_02500000',map_location='cpu',weights_only=True)['generator'],strict=True)
        self.net.eval();self.net.remove_weight_norm()
        self.metadata=dict(vocoder_repository=REPO,vocoder_weight='g_02500000',vocoder_code_commit=COMMIT,
                           mel_convention_warning='Acoustic STFT uses center=True; original HiFi-GAN frontend uses a different padding convention. Same saved comparison setup retained.')

    @torch.inference_mode()
    def __call__(self, mel):
        mel=np.asarray(mel,dtype='float32')
        if mel.ndim!=2 or mel.shape[1]!=80 or len(mel)<1 or not np.isfinite(mel).all(): raise ValueError('Invalid mel')
        wave=self.net(torch.from_numpy(mel.T.copy())[None].to(self.device))[0,0].float().cpu().numpy()
        if len(wave)!=len(mel)*256 or not np.isfinite(wave).all(): raise ValueError('Invalid vocoder waveform')
        return wave
