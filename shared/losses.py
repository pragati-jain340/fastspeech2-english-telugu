import torch
from torch.nn import functional as F
from .model import padmask

def loss_for(output, batch, frame_level):
    token_valid = ~padmask(batch['token_lengths'], batch['tokens'].shape[1])
    mel_valid = ~padmask(batch['mel_lengths'], batch['mel'].shape[1])
    valid = mel_valid if frame_level else token_valid
    pitch = batch['frame_pitch'] if frame_level else batch['pitch']
    energy = batch['frame_energy'] if frame_level else batch['energy']
    voiced = batch['frame_voiced'] if frame_level else batch['voiced']
    mel = F.l1_loss(output['mel'][mel_valid], batch['mel'][mel_valid])
    total = mel + F.mse_loss(output['duration'][token_valid], torch.log1p(batch['duration'][token_valid].float()))
    total = total + (output['pitch'] - pitch).square().masked_fill(~(valid & voiced), 0).sum() / (valid & voiced).sum().clamp_min(1)
    total = total + F.mse_loss(output['energy'][valid], energy[valid])
    total = total + F.binary_cross_entropy_with_logits(output['voicing'][valid], voiced[valid].float())
    return (total, mel)
