import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.model import PilotFS2, padmask
from shared.losses import loss_for
from shared.frontend import tokens_for, encode
from shared.features import validate_feature


@pytest.mark.parametrize('frame_level',[False,True])
def test_padding_duration_and_masked_losses(frame_level):
    torch.set_num_threads(2);torch.manual_seed(7)
    m=PilotFS2(8,frame_level,width=16).eval()
    batch=dict(tokens=torch.tensor([[2,3,4],[2,0,0]]),token_lengths=torch.tensor([3,1]),
        duration=torch.tensor([[1,2,1],[2,0,0]]),mel_lengths=torch.tensor([4,2]),
        mel=torch.randn(2,4,80),pitch=torch.zeros(2,3),energy=torch.zeros(2,3),voiced=torch.ones(2,3,dtype=torch.bool),
        frame_pitch=torch.zeros(2,4),frame_energy=torch.zeros(2,4),frame_voiced=torch.ones(2,4,dtype=torch.bool))
    out=m(batch['tokens'],batch['token_lengths'],targets=batch)
    loss,mel=loss_for(out,batch,frame_level)
    batch['mel'][1,2:]=999999
    assert torch.equal(mel,loss_for(out,batch,frame_level)[1])
    assert torch.isfinite(loss)
    assert torch.equal(out['mel_lengths'],batch['duration'].sum(1))
    free=m(batch['tokens'],batch['token_lengths'],max_frames=1000)
    duration=torch.expm1(free['duration'].clamp(0,math.log1p(100))).round().long().clamp_min(1).masked_fill(padmask(batch['token_lengths'],3),0)
    assert torch.equal(duration.sum(1),free['mel_lengths'])
    with pytest.raises(ValueError): m(batch['tokens'],batch['token_lengths'],max_frames=0)


def test_frontend_numbers_and_unknowns():
    assert ''.join(tokens_for("The orchestra's conduct.",'english'))=="theorchestra'sconduct"
    with pytest.raises(ValueError,match='Digits'): tokens_for('10:45','english')
    with pytest.raises(ValueError,match='absent'): encode('z','english',{'vocabulary':{'a':2}})


def test_feature_rejects_bad_duration():
    a=dict(tokens=np.array(['a']),mel=np.zeros((2,80)),duration=np.array([1]),pitch=np.zeros(1),energy=np.zeros(1))
    with pytest.raises(ValueError,match='duration'): validate_feature(a)


def test_resume_restores_optimizer_rng_and_total_epochs(tmp_path):
    from shared.training import train
    torch.set_num_threads(2)
    torch.manual_seed(91)
    def item():
        return dict(tokens=torch.tensor([2,3]),mel=torch.randn(3,80),duration=torch.tensor([1,2]),
                    pitch=torch.zeros(2),energy=torch.zeros(2),voiced=torch.ones(2,dtype=torch.bool),
                    frame_pitch=torch.zeros(3),frame_energy=torch.zeros(3),frame_voiced=torch.ones(3,dtype=torch.bool))
    items={'train':[item() for _ in range(3)],'validation':[item() for _ in range(3)]}
    meta=dict(vocabulary={'a':2,'b':3},pad_id=0,unk_id=1,pitch_mean=200.,pitch_std=40.,energy_mean=30.,energy_std=20.)
    cfg=dict(seed=42,batch_size=2,max_epochs=2,sample_every=50)
    device=torch.device('cpu');variants=[('frame_prosody',True)]
    train(items,meta,tmp_path/'full',cfg,device,variants)
    train(items,meta,tmp_path/'resume',dict(cfg,max_epochs=1),device,variants)
    train(items,meta,tmp_path/'resume',cfg,device,variants)
    a=torch.load(tmp_path/'full/frame_prosody/last.pt',weights_only=True)
    b=torch.load(tmp_path/'resume/frame_prosody/last.pt',weights_only=True)
    assert a['epoch']==b['epoch']==2
    assert a['updates']==b['updates']==4
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    assert a['latest_row']==b['latest_row']
