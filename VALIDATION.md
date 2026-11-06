# Validation and reproducibility

The following checks were recorded locally on CPU. They describe what was tested; they do not replace an independent Colab execution or a listening evaluation.

## Recorded checks

| Check | Result | Submitted evidence |
|---|---|---|
| Unit tests | 6 passed: masks, target grids, frontend/features, synthetic resume and notebook validity | [tests/](tests/) |
| Current acoustic checkpoints | All 8 loaded with safe loading and strict state dictionaries; metadata and CSV epochs agree | [Verification summary](evidence/verification_summary.json) |
| Refactored computation | 20 predictions bitwise equal to the original implementation on CPU | [Prediction comparison](evidence/inference_validation.json) |
| Saved frame-prosody mels | All ten agree with CPU generation; max absolute difference 0.00004315, within atol 0.005 / rtol 0.001 | [Prediction comparison](evidence/inference_validation.json) |
| Feature extraction | One available raw Telugu clip reproduced identical tokens, durations, mel, pitch and energy | [Preprocessing comparison](evidence/preprocessing_validation.json) |
| New waveform synthesis | English 2.473 s and Telugu 3.471 s; finite mel/audio with expected sample counts | [Verification summary](evidence/verification_summary.json) |
| Assignment deliverables | 10 WAVs, prediction NPZ/JSON files and 30 variance plots verified | [Generation inventory](Results/generation_inventory.csv) |
| Splits and cached targets | Zero source-record overlap; complete unique index coverage for the checked frame caches | [Verification summary](evidence/verification_summary.json) |
| Colab notebooks | Five valid notebooks; code cells compile | [notebooks/](notebooks/) |
| Setup failures | Missing backups, training features and inference checkpoints report a clear next action | CLI checks; START_HERE.md instructions |

## Re-run included tests

From the repository's project root:

```bash
python -m compileall -q shared English Telugu scripts tests
python -m pytest tests -q
```

To evaluate the trained models, follow [START_HERE.md](START_HERE.md). This needs only the submitted code and linked checkpoint backups; the original development workspace is not required.

## Still pending

| Limitation | Effect on evaluation |
|---|---|
| Fresh end-to-end Colab run not recorded | Local checks do not guarantee cloud-runtime compatibility |
| Drive anonymous access not verified | Evaluator download permissions must be confirmed |
| Human listening review incomplete | No unsupported pronunciation, intelligibility or MOS claims are made |
| Original feature ZIPs / some alignment manifests not fully distributed | Exact historical full-run resume cannot yet be verified |
| Historical dataset revision unavailable | Fresh dataset downloads record a new revision, not a proven identical old snapshot |

The one-clip feature check and tiny synthetic resume test are limited checks, not full-dataset reproduction. Correct duration totals do not prove phonetic boundaries. Colab setup preserves the runtime's existing torch/torchaudio versions; the recorded local environment is not a clean Colab replica.
