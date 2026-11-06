import os
import torch

def save_checkpoint(path, state):
    temp = path.with_suffix('.partial')
    torch.save(state, temp)
    os.replace(temp, path)

def load_checkpoint(path, device='cpu', language=None, variant=None):
    from .model import PilotFS2
    from .legacy_model import FastSpeech2
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cfg = saved['model_config']
    actual_variant = ('frame_prosody' if cfg['frame_level'] else 'character_baseline') if 'frame_level' in cfg else 'original_baseline'
    if variant and variant != actual_variant:
        raise ValueError(f'Variant mismatch: requested {variant}, saved {actual_variant}')
    meta = saved['metadata']
    if meta['pad_id'] != 0 or meta['unk_id'] != 1 or cfg['vocab_size'] != len(meta['vocabulary']) + 2:
        raise ValueError('Invalid vocabulary or reserved token IDs')
    if sorted(meta['vocabulary'].values()) != list(range(2,cfg['vocab_size'])):
        raise ValueError('Noncontiguous vocabulary')
    for key in ('pitch_std', 'energy_std'):
        if not meta[key] > 0: raise ValueError(f'Invalid {key}')
    # Historical files lack a language field. This assignment's two vocabularies
    # are distinct; imported new runs carry explicit language metadata.
    actual_language = saved.get('language')
    if not actual_language:
        v=meta['vocabulary']
        actual_language = 'english' if set(v)==set("abcdefghijklmnopqrstuvwxyz'") else 'telugu' if set(v)==set('abcdeghijklmnoprstuvy') else None
    if not actual_language or (language and actual_language != language.lower()):
        raise ValueError('Checkpoint language is missing or does not match the requested language')
    model = (PilotFS2(**cfg) if 'frame_level' in cfg else FastSpeech2(**cfg)).to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    print(f"Loaded language={actual_language} variant={actual_variant} checkpoint={path} epoch={saved['epoch']} updates={saved.get('updates','not recorded')} device={device}")
    return model, saved
