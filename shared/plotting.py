from pathlib import Path
import numpy as np


def variance_plots(root, name, arrays, frame_level):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);d=arrays['duration_frames'];tokens=arrays['tokens']
    f0=arrays['f0_hz'];v=arrays['voiced'];e=arrays['energy_original_units']
    if not frame_level: f0,v,e=(np.repeat(a,d) for a in (f0,v,e))
    times=np.arange(len(f0))*256/22050
    cases=[('duration',np.arange(len(d)),d,'Character position','Mel frames'),
           ('pitch',times,np.where(v&(f0>0),f0,np.nan),'Time (seconds)','F0 (Hz); gaps = unvoiced/invalid'),
           ('energy',times,e,'Time (seconds)','STFT-magnitude L2 energy')]
    for kind,x,y,xlabel,ylabel in cases:
        p=root/kind/f'{name}.png';p.parent.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(13,4))
        if kind=='duration':
            ax.bar(x,y);step=max(1,int(np.ceil(len(tokens)/65)));ax.set_xticks(x[::step],tokens[::step],rotation=90,fontsize=8)
        else: ax.plot(x,y,lw=1.2)
        ax.set(title=f'{name}: predicted character durations' if kind=='duration' else f'{name}: predicted {kind}',xlabel=xlabel,ylabel=ylabel)
        ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(p,dpi=160);plt.close(fig)


def loss_plots(results):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=Path(results)
    for split,column in [('training','train_mel_mae'),('validation','val_mel_mae')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for variant in ('character_baseline','frame_prosody'):
            p=results/f'{variant}_metrics.csv'
            if not p.exists(): continue
            with p.open() as f: rows=list(csv.DictReader(f))
            ax.plot([int(r['updates']) for r in rows],[float(r[column]) for r in rows],label=variant)
        ax.set(title=f'{split.title()} mel MAE (recorded prosody)',xlabel='Optimizer updates',ylabel='Log-mel MAE')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(results/f'{split}_loss.png',dpi=160);plt.close(fig)
