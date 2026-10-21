# SCFP V2: Semantic Slots + Adaptive Feature-level DFT

Post-training audit (no retraining): [Vietnamese audit report and Kaggle commands](SCFP_V2_AUDIT.md),
[diagnostic notebook](../../notebooks/irra_scfp_v2_audit_kaggle.ipynb),
or `python -m experiments.scfp.audit --help`. The audit reads existing V2 weights,
measures actual frequency responses/slot attention/shared-backbone gradients, and
compares frequency ON/OFF on an explicitly labeled fixed training subset.

This is an experimental extension of IRRA, **not an established improvement**.
V3 semantic guidance, teachers and distillation are deliberately absent.

## Lean training notebook: matched IRRA comparison

Use [`irra_scfp_v2_train_kaggle.ipynb`](../../notebooks/irra_scfp_v2_train_kaggle.ipynb) for the user's requested comparison with the original IRRA training run. It has **three code cells**: configuration, git clone/setup, and training. Import the notebook, attach only the original ICFG-PEDES dataset (`ICFG-PEDES.json` and `imgs/`), enable GPU and Internet, then Run All. Setup clones `https://github.com/thanhphu25/IRRA.git`, branch `feat/scfp-v2`, and prints the cloned commit. That branch must contain the updated launcher before the notebook can run.

Defaults: V2, **60 epochs**, batch 64, backbone LR 1e-5, SCFP LR 1e-4, seed **1** as in original IRRA. The repository supplies the original ViT-B/16 architecture (384×128), Adam optimizer and LR multipliers, SDM/MLM/ID losses, cosine schedule and five warmup epochs; the notebook enables augmentation like `run_irra.sh`. Fresh training needs no old `configs.yaml`, trained checkpoint, source ZIP, or `processed_data/*.pkl`. The launcher still accepts an optional baseline config for users who explicitly want to import custom settings. New runs initialize from pretrained CLIP. `configs.yaml` is generated in the new run's **output** directory for provenance and resume. Set `EXPERIMENT="baseline"` to rerun the control under the same setup.

This explicitly selected `--training_protocol irra` uses **all official training identities**, evaluates official-test R1/R5/R10 each epoch, and selects `best.pth` by official-test R1, following the original IRRA pipeline. It is a paper-style comparison protocol, **not independent validation**; metadata labels the selection split as `test`. The separate `scientific` protocol still requires a pre-training PID holdout. Do not describe test-based checkpoint selection as unseen validation or tune additional hyperparameters on those scores.

The lean notebook skips unit tests, smoke runs and fixed-sample diagnostics, saves `best.pth` / `last.pth` with configs and logs, and computes exact R1/R5/R10/mAP/mINP after training. Outputs stay under `/kaggle/working/SCFP_V2_train`; checkpoints are not duplicated into an automatic ZIP. Set `RESUME_CHECKPOINT` to the original run's last.pth to continue its saved total budget and schedule. Resume also works when only `last.pth` is attached: the launcher restores its embedded configuration. For a run mounted in read-only Kaggle Input, only checkpoints and matching run artifacts are copied into Working, with writable copies for appending logs. Unrelated image/code trees are not staged. Baseline config is not needed for resume; existing Input files remain untouched.

Training logs stream into the notebook, and the output directory is printed before launching. Interrupting the training cell stops its child process group, including DataLoader workers; resume continues after the last completed epoch saved in last.pth. The launcher works independently of the notebook's current directory. Setup clears cached modules from the previous source on rerun.

The cloned branch must include `experiments/scfp/kaggle_train.py` and `experiments/scfp/requirements-train.txt`; setup reports a clear error if the latest notebook code has not been published yet. The notebook is prepared and tested locally using a local Git clone fixture and a toy CPU training/evaluation/resume pipeline; actual Kaggle T4 execution and the 60-epoch real-data run have not been executed here. The older engineering notebook below retains its optional ZIP-source workflow.

## Repository audit

- Branch: `feat/scfp-v2`, created directly from updated `main`.
- Base: `cfc23785545ead82dee2749fa0fc0ad948c96430`.
- `feat/lgsp-tbpr` was read only. No merge/cherry-pick was performed.
- Infrastructure identified: strict checkpoint loading, chunked metrics, train-only PID holdout, Kaggle discovery/symlinks, diagnostics, smoke testing and resumable logging.
- Actually transferred: `utils/metrics.py`, `datasets/splits.py`, Kaggle preparation and metric regression tests. These were reviewed/adapted. Checkpoint handling, smoke testing, logging and the notebook were rewritten for V2, informed by that audit.
- Excluded: `model/lgsp/`, RGB DFT, convolutional prompt pool, spatial fusion, LGSP wrappers, classifier/session/FSCIL logic, and old `lgsp_mode` configuration behavior.
- `prompt.md` was already an untracked user file. Only the requested branch name was corrected; it remains uncommitted.

## Implemented architecture

Image: CLIP patch embedding → visual blocks 1–6 → adaptive feature FFT → remaining blocks → projected global CLS + patch tokens. The default ViT-B/16 input is `[B,3,384,128]`, with `[B,193,768]` at injection and final `[B,193,512]`. The actual convolution output provides the rectangular grid; overlapping strides are supported. An injection before the last block is required so later attention propagates changed patches into CLS. The baseline keeps the original Transformer call and parameter names.

The FFT operates over **spatial axes only**, in FP32 with autocast disabled. Four configurable, nonoverlapping hard radial bands cover all unshifted FFT bins. Coordinates are normalized cycles per pixel independently along each axis. Magnitude-symmetric masks preserve conjugate symmetry, including odd grid sizes and Nyquist bins; inverse output uses `.real`, never `abs()`.

Gate: global spatial average → LayerNorm → MLP(768→64→4) → softmax(logits/temperature). For each image and band,

```text
alpha = tanh(alpha_raw), initially 0.01
G = 1 + alpha * sum_r tanh(R * weight_r - 1) * mask_r
```

Thus `0 <= G <= 2` numerically, with a near-identity initialization. The extra tanh relative to the prompt's suggested response bounds gains even if the band count or logits grow. Both MLP layers start nonzero; alpha is nonzero by default. Setting alpha exactly zero is safe but temporarily blocks gate gradients until alpha moves. CLS is unchanged **at injection**, then changes through the subsequent ViT blocks. Train, MLM and image inference share this pathway.

Text: normal CLIP causal encoder → global EOT + projected token features. No text FFT. The tokenizer was inspected: padding=0, MASK=49405, SOT=49406, EOT=49407. Content masking excludes special tokens and all positions from EOT onward. Empty descriptions use a zero sentinel, avoiding all-masked attention without treating padding as content.

Slots: shared orthogonal learnable prototypes `[K,512]`, separate visual/text query projections and multihead cross attention, LayerNorm, residual prototypes, lightweight FFN, L2 normalization. Defaults: K=4, heads=4, dropout=0. Text relevance uses `0.1 + 0.9*sigmoid(head(slot))`, then sum normalization; all relevance values cannot vanish. This is an inductive bias, not supervised attribute presence.

Local scores use corresponding-index slot cosine similarities, aggregated by normalized query-text relevance. A `[B,B]` bidirectional local SDM uses **all equal-PID positives**, including repeated identities. Its distribution formula follows original IRRA SDM, with dimension-safe target normalization. Squared off-diagonal cosine diversity discourages collapse. Shared slots + local supervision encourage correspondence; they do not prove semantic slot correspondence.

```text
L = original SDM + original weighted ID + original weighted MLM
    + 0.1 * local_SDM + 0.01 * mean(visual_diversity, text_diversity)
```

Original `model/objectives.py` is unchanged. Auxiliary losses are returned already weighted, and the processor sums them once. Both auxiliary weights can be zero. Default retrieval remains independent image/text global embeddings; gallery vectors remain precomputable. Slots do not run in the default inference API.

| Mode | Slots/local loss | Intermediate FFT |
|---|---|---|
| `baseline` (CLI default) | off | off |
| `slots` | on | off |
| `frequency` | off | on |
| `v2` | on | on |

## Paper-to-code mapping

- [IRRA, CVPR 2023](https://arxiv.org/abs/2303.12501), [source](https://github.com/anosorae/IRRA): existing CLIP encoders, global embeddings, SDM/ID/MLM and train/evaluation structure are retained. No new objective replaces global SDM.
- [PLOT, ECCV 2024](https://arxiv.org/abs/2409.13475), [slot code](https://github.com/jicheol93/PLOT/blob/main/model/slot_model.py), [model code](https://github.com/jicheol93/PLOT/blob/main/model/build_local_slot_detachRecon.py): conceptual transfer of shared learnable part slots and text-dependent aggregation. PLOT uses iterative competitive slot attention/GRU and reconstruction; V2 implements a new single-stage multihead module, no reconstruction, and auxiliary local supervision rather than local retrieval scoring.
- [LGSP-Prompt, ICCV 2025](https://arxiv.org/abs/2507.09183), [frequency implementation](https://github.com/Jywsuperman/LGSP/blob/main/models/base/ViT_Network.py), [prompt code](https://github.com/Jywsuperman/LGSP/blob/main/models/prompt.py): radial Fourier selection inspired the new module. V2 uses intermediate CLIP features, image-conditioned gates, complete hard bands, signed real inverse reconstruction and bounded near-identity gains. No LGSP architecture source was copied.
- [EESSO, Image and Vision Computing 2024](https://doi.org/10.1016/j.imavis.2024.104912): publisher abstract/highlights describe high/low frequency decomposition and uncertainty-guided multi-stream learning for TBPR. Full text/source was not available in this session. Frequency learning in TBPR is therefore not claimed as new.
- [ITSELF](https://trhuuloc.github.io/itself/): project-page concepts of attention-guided fine-grained alignment/patch selection were reviewed. No dependency or reproduction is introduced; project-page performance claims are not independently verified here.

All files in `model/scfp/` are newly written experimental code, conceptually adapted from the above mechanisms. They are not reproductions of PLOT or LGSP.

## Source changes

| File | Purpose |
|---|---|
| `model/scfp/feature_frequency.py` | Spatial FFT masks, bounded residual, feature/diagnostic interface |
| `model/scfp/frequency_gate.py` | Image-dependent FP32 MLP gate |
| `model/scfp/semantic_slots.py` | Shared prototypes, dual-modality slots, text masks and relevance |
| `model/scfp/local_alignment.py` | Local similarities and repeated-PID bidirectional SDM |
| `model/scfp/diversity.py` | Optional slot diversity penalty |
| `model/scfp/__init__.py` | Module package |
| `model/clip_model.py` | Minimal intermediate adapter; offline CLIP construction from IRRA checkpoint shapes |
| `model/build.py` | Four modes, auxiliary losses, optional features, CPU-safe classifier dtype and freezing |
| `solver/build.py` | Exactly one group per parameter; SCFP LR takes precedence over bias/cross rules |
| `solver/lr_scheduler.py` | Avoid division by zero for short total/warmup schedules |
| `datasets/splits.py` | Deterministic identity-disjoint holdout, preserving classifier indices |
| `datasets/build.py` | Holdout manifest, no-evaluation mode, automatic MLM dataset, distributed loader fixes |
| `utils/options.py` | SCFP settings, YAML presets, protocol guards, warm-start/resume distinction |
| `utils/checkpoint.py` | Strict shape/key/metadata loading, offline-compatible warm-start, resume and provenance |
| `utils/metrics.py` | Exact query-chunked R1/R5/R10/mAP/mINP, explicit missing-positive warning |
| `utils/logger.py` | Append logs when resuming |
| `utils/iotools.py` | Save safe YAML presets that can be read back as resume configuration |
| `train.py` | Fixed seed, separate init/resume, unique runs, environment metadata |
| `processor/processor.py` | Original loss loop extended with logs, diagnostics, last/best checkpoint semantics |
| `test.py` | Reconstruct checkpoint mode, explicit path overrides, exact final metrics export |
| `experiments/scfp/prepare_kaggle.py` | Unique dataset/checkpoint discovery and safe symlinks; CLI |
| `experiments/scfp/smoke_test.py` | One-batch real/tiny forward/backward, near-identity and checkpoint checks |
| `experiments/scfp/diagnostics.py` | Gradient/slot/frequency statistics, fixed samples, environment/source provenance |
| `experiments/scfp/bundle.py` | Package current working source without pushing; source hash manifest |
| `experiments/scfp/kaggle_train.py` | Dataset-only IRRA comparison defaults, optional config import, full training, final metrics and resume |
| `experiments/scfp/configs/{baseline,slots,frequency,v2}.yaml` | Four CLI-loadable ablation presets |
| `experiments/scfp/requirements.txt` | Lightweight modern dependencies; retain Kaggle CUDA torch/torchvision |
| `experiments/scfp/requirements-train.txt` | Training dependencies without pytest/nbformat |
| `experiments/scfp/validation_report.md` | Actual observed tests and outstanding Kaggle checks |
| `experiments/scfp/reports/cpu_tiny_smoke.json` | Raw observed synthetic CPU smoke diagnostics |
| `notebooks/irra_scfp_v2_icfg_kaggle.ipynb` | Single-GPU setup, source ZIP/branch, smoke/train/evaluation/export |
| `notebooks/irra_scfp_v2_train_kaggle.ipynb` | Three-cell Git clone and dataset-only, 60-epoch from-CLIP training |
| `tests/test_feature_frequency.py` | Grid, dtype, symmetry, adaptivity, initialization and gradient checks |
| `tests/test_semantic_slots.py` | Slot shapes, masks, short/empty text, relevance and gradients |
| `tests/test_local_alignment.py` | Repeated PID targets, SDM equivalence and degenerate batches |
| `tests/test_model_integration.py` | All four modes, losses, optimizer, embeddings and roundtrip |
| `tests/test_checkpoint_compatibility.py` | Original-main regression, warm-start, corruption, metadata, RNG resume |
| `tests/test_metrics.py` | Exact full/chunked equivalence and missing positives |
| `tests/test_workflow.py` | Protocol guards, presets, split, Kaggle utilities and train/save/resume |
| `tests/test_notebook.py` | Notebook schema/syntax/defaults and source bundle |
| `tests/test_kaggle_train.py` | Matched config/protocol, lean notebook, actual toy CLI training/evaluation/resume |
| `tests/test_cuda.py` | CUDA autocast/FP16 tests, skipped when CUDA unavailable |
| `tests/helpers.py`, `tests/conftest.py`, `tests/__init__.py` | Offline small-CLIP fixtures and test setup |
| `tests/fixtures/irra_build_main.py` | Unmodified baseline snapshot from the base commit for regression |
| `.gitignore`, `pytest.ini` | Exclude generated weights/runs/caches and scope test discovery |
| `THIRD_PARTY_NOTICES.md`, root `README.md` | Attribution and V2 entry point |

## Checkpoints and optimization

`--init_checkpoint` loads compatible original IRRA weights only, reporting every missing/unexpected/shape mismatch. Only missing V2 keys are permitted. Backbone, ID and MLM mismatches fail. Warm-start does **not** restore the old optimizer, scheduler, epoch or best score. Offline construction infers CLIP architecture from the checkpoint; warm-start/resume/evaluation need no second OpenAI model download.

Trained V2 loading is strict, including metadata such as insertion block, gate temperature, mode, image size, stride and losses. A partial V2 checkpoint, even if all V2 tensors were removed, fails when its metadata identifies it as V2. Plain tensor-only state dictionaries cannot supply architecture metadata; use matching configuration explicitly.

`--resume_checkpoint` restores model, optimizer, scheduler, saved best score and RNG. It starts at saved epoch + 1. Keep the original total epoch budget/split/schedule/auxiliary loss settings. Extending a completed 1-epoch schedule to 5 epochs is a **new** experiment, not a transparent resume. Batch-boundary/worker-prefetch or non-deterministic GPU kernels may still prevent bitwise reproduction; checkpointing occurs at epoch boundaries.

New parameters stay FP32. Original modules follow IRRA FP16 by default; CPU uses FP32. FFT/gate/slot math explicitly disables autocast. Existing optimizer head/bias multipliers remain; all SCFP parameters use `--scfp_lr` (default 1e-4). `--freeze_backbone` freezes original CLIP parameters while keeping frequency and slot parameters trainable.

## Kaggle: exact commands

Use the notebook for automatic discovery and original-config import. Internet is needed for lightweight dependencies and original CLIP initialization in scientific mode; use mounted/cache CLIP weights via `--pretrain_choice /path/to/CLIP.pt` if needed. Warm-start reconstructs CLIP directly from `best.pth`.

1. Package local uncommitted source, attach the ZIP to Kaggle Input, then open the notebook:

```bash
python -m experiments.scfp.bundle --output /tmp/IRRA_scfp_v2_source.zip
```

The branch has not been pushed automatically. The notebook can clone `feat/scfp-v2` only after the user publishes it; source ZIP works without that step.

2. Prepare mounted data (ambiguous sources fail and request explicit paths):

```bash
python -m pip install -r experiments/scfp/requirements.txt
python -m experiments.scfp.prepare_kaggle --data_root /kaggle/working/data
```

3. Attach the user's original checkpoint dataset. In a Kaggle terminal:

```bash
SCFP_CHECKPOINT=/kaggle/input/datasets/tphu25/irra-icfgpedes-checkpoint/IRRA_runs/ICFG-PEDES/20261008_162321_irra_icfg_paper/best.pth
SCFP_ORIGINAL_CONFIG=/kaggle/input/datasets/tphu25/irra-icfgpedes-checkpoint/IRRA_runs/ICFG-PEDES/20261008_162321_irra_icfg_paper/configs.yaml
```

The notebook verifies these through discovery rather than assuming the mounted slug. CLI examples below assume the documented ViT-B/16, 384×128, stride 16 and cmt_depth=4 configuration; use actual original config values if different.

4. Unit tests and the first **full V2 warm-start smoke**:

```bash
python -m pytest -q
python -m experiments.scfp.smoke_test \
  --scfp_mode v2 --training_protocol warmstart --val_dataset none \
  --config_file "$SCFP_ORIGINAL_CONFIG" --init_checkpoint "$SCFP_CHECKPOINT" \
  --dataset_name ICFG-PEDES --root_dir /kaggle/working/data \
  --batch_size 64 --num_workers 2 --output_file /kaggle/working/v2_smoke.json
```

Optional official baseline regression (explicit evaluation, never automatic selection):

```bash
python test.py --config_file "$SCFP_ORIGINAL_CONFIG" --checkpoint "$SCFP_CHECKPOINT" \
  --root_dir /kaggle/working/data --test_batch_size 64 \
  --metrics_output /kaggle/working/baseline_regression.json
```

Historical **user-reported** baseline: R1=63.573%, R5=80.416%, R10=85.868%, mAP=38.149%, mINP=7.919%. Full evaluation was not run locally. No threshold silently certifies these values.

5. V2 one-epoch engineering run:

```bash
python train.py --experiment_config experiments/scfp/configs/v2.yaml \
  --training_protocol warmstart --val_dataset none --init_checkpoint "$SCFP_CHECKPOINT" \
  --dataset_name ICFG-PEDES --root_dir /kaggle/working/data \
  --img_size 384 128 --stride_size 16 --img_aug --MLM --loss_names sdm+mlm+id \
  --batch_size 64 --num_workers 2 --seed 42 --num_epoch 1 --warmup_epochs 0 \
  --lr 1e-6 --scfp_lr 1e-4 --scfp_diagnostics --name V2_full \
  --output_dir /kaggle/working/SCFP_V2_runs/V2_full
```

6. Five exploratory epochs: repeat the preceding command with `--num_epoch 5 --warmup_epochs 1 --name V2_explore5`. Initialize from the same original checkpoint, write a new unique run. This is not an automatic four-mode sweep and is not evidence of generalization.

7. Controlled four-mode scientific screening, **run each explicitly**:

```bash
SCFP_MODE=baseline  # repeat separately: slots, frequency, v2
python train.py --experiment_config "experiments/scfp/configs/$SCFP_MODE.yaml" \
  --training_protocol scientific --val_dataset holdout --validation_fraction 0.1 \
  --pretrain_choice ViT-B/16 --dataset_name ICFG-PEDES --root_dir /kaggle/working/data \
  --img_size 384 128 --stride_size 16 --img_aug --MLM --loss_names sdm+mlm+id \
  --batch_size 64 --num_workers 2 --seed 42 --num_epoch 15 --warmup_epochs 3 \
  --lr 1e-5 --scfp_lr 1e-4 --scfp_diagnostics --name "$SCFP_MODE" \
  --output_dir "/kaggle/working/SCFP_V2_runs/$SCFP_MODE"
```

Compare saved `split_manifest.json` files: identical PID membership is required. Keep augmentation/batch size/seed/loss weights/schedule/budget fixed. Classifier indices are preserved from the original training set, leaving unused holdout-class outputs but **no holdout examples** in training. Ten–fifteen epochs are only screening; convergence is not guaranteed. Consider 30 epochs, then 60 epochs and 3 seeds only after deciding to proceed. Official test is reserved for a decided configuration.

8. Resume an interrupted run with its exact saved settings:

```bash
SCFP_LAST=/kaggle/working/SCFP_V2_runs/V2_full/ICFG-PEDES/ACTUAL_RUN/last.pth
python train.py --experiment_config "$(dirname "$SCFP_LAST")/configs.yaml" \
  --init_checkpoint '' --resume_checkpoint "$SCFP_LAST"
```

This reads the original args, including total epochs; the explicit empty init flag avoids combining warm-start and resume. `last.pth` is saved each epoch. Training logs append. `best.pth` is selected only when validation exists; warm-start without validation saves **last only**, with null best score. Resuming a finished budget performs no extra training.

9. Final official evaluation and export (after scientific configuration selection):

```bash
SCFP_RUN=/kaggle/working/SCFP_V2_runs/v2/ICFG-PEDES/ACTUAL_RUN
python test.py --config_file "$SCFP_RUN/configs.yaml" --checkpoint "$SCFP_RUN/best.pth" \
  --root_dir /kaggle/working/data --test_batch_size 64 --chunk_size 128 \
  --metrics_output "$SCFP_RUN/final_test_metrics.json"
python -m zipfile -c /kaggle/working/SCFP_V2_export.zip "$SCFP_RUN"
```

Metrics exporters refuse to overwrite existing JSON. Training creates unique directories. The notebook exports only the selected run plus session checks. Typical outputs: configs/environment/source provenance, train_log.txt, training_stats.jsonl, validation_metrics.jsonl, metrics.json, diagnostics.json, fixed sample tensor/maps, last.pth, and best.pth when validation runs. Exact final mAP/mINP sort the **entire gallery** per query chunk; top-k training metrics do not approximate mAP. Missing gallery positives are warned and reported explicitly.

## Diagnostics / optional features

`model(batch, return_features=True)` returns `(losses, features)` with `frequency_logits [B,R]`, `frequency_weights [B,R]`, `frequency_band_features [B,R,C]` (per-channel RMS band energy), `visual_slots [B,K,D]`, `text_slots`, `slot_relevance` and attention maps. Frequency band energy is optional to avoid normal-training FFT/memory overhead. These are V2 interfaces only, not V3 guidance.

`--scfp_diagnostics` records per-image/mean/std/entropy/variance frequency gates, alpha, FFT imaginary residual, enhancement residual norm/ratio, gradient norms by group, slot similarities, attention entropy, normalized/raw text relevance and local positive/negative margin. A fixed tensor batch is saved at the beginning of the run and reused after each epoch and resume; its attention maps and band energies are in `fixed_sample_features.pt`. No body-part labels are assigned to slots.

## Limits and remaining checks

See [validation_report.md](validation_report.md). Local tests use randomly initialized small CLIP fixtures and synthetic images/captions; they do not demonstrate retrieval quality. CUDA autocast, actual checkpoint/data compatibility, T4 batch-size-64 memory/time, full notebook execution and official metrics must still be verified on Kaggle. Single-GPU SCFP is enforced; distributed SCFP is not validated.

Possible failure modes: gate becomes nearly global (check per-image variance), slots collapse or indices fail to align (check similarity/matching/maps), normalized relevance becomes uninformative, auxiliary gradients dominate, extra memory exhausts T4, and warm-start forgets the converged backbone. An image-only gate and shared prototypes do not establish semantic interpretability. Do not select hyperparameters on the official test or on a holdout already seen by the original checkpoint. No novelty/superiority claim is supported before controlled, converged, multi-seed comparisons to related methods.
