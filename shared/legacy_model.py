"""Original notebook architecture; never substitute PilotFS2 on resume."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def padding_mask(lengths, width):
    return torch.arange(width, device=lengths.device)[None, :] >= lengths[:, None]

def add_positions(x):
    positions = torch.arange(x.size(1), device=x.device).float()
    frequencies = torch.exp(-math.log(10000) * torch.arange(0, x.size(2), 2, device=x.device).float() / x.size(2))
    angles = positions[:, None] * frequencies[None, :]
    encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
    return x + encoding.to(x.dtype)

class FFTBlock(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 2, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.conv_in = nn.Conv1d(width, 2 * width, 3, padding=1)
        self.conv_out = nn.Conv1d(2 * width, width, 3, padding=1)
        self.conv_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, pad):
        mask = pad.unsqueeze(-1)
        attended, _ = self.attention(x, x, x, key_padding_mask=pad, need_weights=False)
        x = self.attention_norm(x + self.dropout(attended)).masked_fill(mask, 0)
        hidden = F.relu(self.conv_in(x.transpose(1, 2)))
        hidden = self.dropout(hidden).masked_fill(pad[:, None, :], 0)
        output = self.conv_out(hidden).transpose(1, 2)
        return self.conv_norm(x + self.dropout(output)).masked_fill(mask, 0)

class VariancePredictor(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(width, 1)

    def forward(self, x, pad):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.norm(F.relu(x)))
        return self.output(x).squeeze(-1).masked_fill(pad, 0)

class FastSpeech2(nn.Module):

    def __init__(self, vocab_size, width=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.encoder = nn.ModuleList([FFTBlock(width) for _ in range(2)])
        self.decoder = nn.ModuleList([FFTBlock(width) for _ in range(2)])
        self.predictors = nn.ModuleDict({key: VariancePredictor(width) for key in ('duration', 'pitch', 'energy', 'voicing')})
        self.pitch_embedding = nn.Linear(1, width)
        self.energy_embedding = nn.Linear(1, width)
        self.mel_output = nn.Linear(width, 80)

    def forward(self, tokens, token_lengths, targets=None):
        token_pad = padding_mask(token_lengths, tokens.size(1))
        x = add_positions(self.embedding(tokens)).masked_fill(token_pad[..., None], 0)
        for block in self.encoder:
            x = block(x, token_pad)
        pred = {key: head(x, token_pad) for key, head in self.predictors.items()}
        if targets is None:
            durations = torch.expm1(pred['duration'].clamp(0, math.log1p(100))).round().long().clamp_min(1).masked_fill(token_pad, 0)
            pitch, energy = (pred['pitch'], pred['energy'])
            voiced = (pred['voicing'] >= 0) & ~token_pad
        else:
            durations = targets['duration']
            pitch, energy, voiced = (targets['pitch'], targets['energy'], targets['voiced'])
        pitch_condition = self.pitch_embedding(pitch.unsqueeze(-1))
        pitch_condition = pitch_condition.masked_fill(~voiced.unsqueeze(-1), 0)
        x = x + pitch_condition + self.energy_embedding(energy.unsqueeze(-1))
        expanded = nn.utils.rnn.pad_sequence([torch.repeat_interleave(row, counts, dim=0) for row, counts in zip(x, durations)], batch_first=True)
        mel_lengths = durations.sum(dim=1)
        mel_pad = padding_mask(mel_lengths, expanded.size(1))
        x = add_positions(expanded).masked_fill(mel_pad[..., None], 0)
        for block in self.decoder:
            x = block(x, mel_pad)
        return {**pred, 'mel': self.mel_output(x), 'mel_lengths': mel_lengths, 'used_duration': durations}

def losses_for(pred, batch):
    valid = batch['tokens'].ne(0)
    mel_valid = ~padding_mask(batch['mel_lengths'], batch['mel'].size(1))
    voiced = valid & batch['voiced']
    pitch_error = (pred['pitch'] - batch['pitch']).square()
    losses = {'mel': F.l1_loss(pred['mel'][mel_valid], batch['mel'][mel_valid]), 'duration': F.mse_loss(pred['duration'][valid], torch.log1p(batch['duration'][valid].float())), 'pitch': pitch_error.masked_fill(~voiced, 0).sum() / voiced.sum().clamp_min(1), 'energy': F.mse_loss(pred['energy'][valid], batch['energy'][valid]), 'voicing': F.binary_cross_entropy_with_logits(pred['voicing'][valid], batch['voiced'][valid].float())}
    losses['total'] = sum(losses.values())
    return losses
