# English checkpoints

**[Download trained weights](https://drive.google.com/file/d/1PkGkmCfh1C-CWESDjNyl2X6fdpA-Glz-/view?usp=sharing)** ? [Colab setup](../../START_HERE.md#1-download-checkpoints)

| File | Use |
|---|---|
| best_mel.pt | Best-validation inference |
| last.pt | Latest optimizer/model state for resume |
| metrics.csv | Training and validation history |
| manifest.json (included here) | Checkpoint epochs, configuration and hashes |

1. Download and extract the backup into Drive.
2. Set BACKUP_DIR to its parent folder; run the notebook's import cell.
3. Weights are imported into DATA_ROOT/runs/english/.

```text
runs/english/
??? frame_prosody/{best_mel.pt,last.pt,metrics.csv}
??? character_baseline/{best_mel.pt,last.pt,metrics.csv}
```

Download permissions remain unverified. Exact training resume additionally requires matching feature/target caches. Large weights are distributed through the link, not bundled here.
