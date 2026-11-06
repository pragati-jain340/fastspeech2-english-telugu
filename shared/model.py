"""Exact PilotFS2 computation extracted from the fixed English training script."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def padmask(lengths, size):
    return torch.arange(size, device=lengths.device)[None] >= lengths[:, None]

def positions(x):
    t = torch.arange(x.shape[1], device=x.device).float()
    f = torch.exp(-math.log(10000) * torch.arange(0, x.shape[2], 2, device=x.device).float() / x.shape[2])
    a = t[:, None] * f[None]
    return x + torch.stack((a.sin(), a.cos()), dim=-1).flatten(1).to(x.dtype)

class Block(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 2, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.conv_in = nn.Conv1d(width, 2 * width, 3, padding=1)
        self.conv_out = nn.Conv1d(2 * width, width, 3, padding=1)
        self.conv_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, pad):
        m = pad.unsqueeze(-1)
        a, _ = self.attention(x, x, x, key_padding_mask=pad, need_weights=False)
        x = self.attention_norm(x + self.dropout(a)).masked_fill(m, 0)
        h = F.relu(self.conv_in(x.transpose(1, 2)))
        h = self.dropout(h).masked_fill(pad[:, None, :], 0)
        return self.conv_norm(x + self.dropout(self.conv_out(h).transpose(1, 2))).masked_fill(m, 0)

class Predictor(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(width, 1)

    def forward(self, x, pad):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(self.norm(F.relu(h)))
        return self.output(h).squeeze(-1).masked_fill(pad, 0)

class PilotFS2(nn.Module):

    def __init__(self, vocab_size, frame_level=False, width=128):
        super().__init__()
        self.frame_level = frame_level
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.encoder = nn.ModuleList([Block(width) for _ in range(2)])
        self.decoder = nn.ModuleList([Block(width) for _ in range(2)])
        self.predictors = nn.ModuleDict({k: Predictor(width) for k in ('duration', 'pitch', 'energy', 'voicing')})
        self.pitch_embedding = nn.Linear(1, width)
        self.energy_embedding = nn.Linear(1, width)
        self.mel_output = nn.Linear(width, 80)

    def forward(self, tokens, lengths, targets=None, max_frames=None):
        token_pad = padmask(lengths, tokens.shape[1])
        x = positions(self.embedding(tokens)).masked_fill(token_pad[..., None], 0)
        for b in self.encoder:
            x = b(x, token_pad)
        duration_pred = self.predictors['duration'](x, token_pad)
        if targets is None:
            duration = torch.expm1(duration_pred.clamp(0, math.log1p(100))).round().long().clamp_min(1).masked_fill(token_pad, 0)
        else:
            duration = targets['duration']
        if max_frames is not None and int(duration.sum(1).max()) > max_frames:
            raise ValueError(f'Predicted {int(duration.sum(1).max())} frames exceeds safety limit {max_frames}')
        if not self.frame_level:
            p = {k: self.predictors[k](x, token_pad) for k in ('pitch', 'energy', 'voicing')}
            if targets is None:
                pitch, energy = (p['pitch'], p['energy'])
                voiced = (p['voicing'] >= 0) & ~token_pad
            else:
                pitch, energy, voiced = (targets['pitch'], targets['energy'], targets['voiced'])
            emb = self.pitch_embedding(pitch.unsqueeze(-1)).masked_fill(~voiced.unsqueeze(-1), 0)
            x = x + emb + self.energy_embedding(energy.unsqueeze(-1))
        x = nn.utils.rnn.pad_sequence([torch.repeat_interleave(row, counts, dim=0) for row, counts in zip(x, duration)], batch_first=True)
        mel_lengths = duration.sum(1)
        mel_pad = padmask(mel_lengths, x.shape[1])
        x = positions(x).masked_fill(mel_pad[..., None], 0)
        if self.frame_level:
            p = {k: self.predictors[k](x, mel_pad) for k in ('pitch', 'energy', 'voicing')}
            if targets is None:
                pitch, energy = (p['pitch'], p['energy'])
                voiced = (p['voicing'] >= 0) & ~mel_pad
            else:
                pitch, energy, voiced = (targets['frame_pitch'], targets['frame_energy'], targets['frame_voiced'])
            emb = self.pitch_embedding(pitch.unsqueeze(-1)).masked_fill(~voiced.unsqueeze(-1), 0)
            x = x + emb + self.energy_embedding(energy.unsqueeze(-1))
        for b in self.decoder:
            x = b(x, mel_pad)
        return dict(duration=duration_pred, **p, mel=self.mel_output(x), mel_lengths=mel_lengths)
