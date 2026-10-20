# Observed validation — 2026-10-20

These are engineering checks on synthetic data, not retrieval results on ICFG-PEDES.

## Environment

- Python 3.12.3 in isolated `/tmp/irra-scfp-env`.
- PyTorch 2.14.1+cpu; torchvision 0.29.1+cpu.
- CUDA unavailable. Actual Kaggle input data and the user's trained checkpoint were not mounted locally.
- Original workspace Python 3.14 did not have torch; its environment was not changed.

## Checks performed

| Check | Status | Observed result |
|---|---|---|
| Automated CPU suite | PASS | 38 passed; 2 CUDA cases skipped; 2 expected missing-positive warnings |
| Frequency P1 before adding slots | PASS | Initial six tests passed, including `[2,193,768]` / 24×8 and an odd rectangular grid |
| FFT autocast precision | PASS | FFT observed FP32 under CPU autocast; FP16 token dtype preserved after reinsertion |
| FFT symmetry / shape / gradients | PASS | CLS unchanged at injection, finite reconstruction, sample-dependent gates, finite nonzero gate/logit gradients |
| Shared visual/text slots | PASS | `[2,4,512]`, normalized output, excluded padding/special/post-EOT content, empty text finite |
| Repeated identity local alignment | PASS | `[1,1,2,3]` has both same-PID positives; both directions finite; one-slot loss agrees with original SDM |
| All four modes | PASS | Forward/backward, original losses, correct auxiliary losses, 512-d embeddings, optimizer registration and save/reload |
| Baseline versus original main source | PASS | Same state-dict keys and matching losses/global embeddings after identical weights; original source snapshot used |
| Warm-start compatibility (fixture) | PASS | Only new SCFP keys missing; no original shape mismatch/unexpected key; optimizer remains fresh |
| Checkpoint corruption / metadata | PASS | Wrong shapes, unexpected keys, missing backbone, partial/new V2 keys and wrong injection layer rejected |
| Resume (fixture) | PASS | Epoch + 1, optimizer/scheduler/RNG, saved zero best score retained |
| Frozen backbone / zero auxiliary weights | PASS | Original CLIP frozen, gate still receives global-loss gradients, auxiliary losses zero, feature interface present |
| Exact chunked metrics | PASS | Full-matrix and chunked R1/R5/R10/mAP/mINP agree on synthetic retrieval; absent positives explicitly warned |
| Notebook schema / code syntax / defaults | PASS | nbformat validation, every Python cell parses, V2/1-epoch/test-off defaults |
| Source bundle | PASS | Includes modules, tests, tokenizer vocabulary, configs and notebook; excludes weights and `.git` |
| Full V2 CPU smoke, small CLIP | PASS | Forward/backward, one optimizer step and checkpoint roundtrip; report linked below |
| Real DataLoader path using toy image files | PASS | CLI smoke used original-style checkpoint/config plus 512-d small CLIP fixture and actual PIL/tokenizer/DataLoader |
| CLI warm-start train | PASS | One epoch limited to **one batch** on toy data; five finite losses, diagnostics/logs and last.pth saved; no best selected |
| CLI scientific train | PASS | One batch on toy data, raw toy CLIP initialization, disjoint PID manifest, validation and best/last checkpoints saved |
| CLI saved-config resume | PASS | Warm-start and scientific runs reconstructed offline and accepted original saved schedule; completed budgets added no epochs |
| CLI evaluation | PASS | Trained toy V2 reconstructed from config/checkpoint; exact metric JSON written on toy test data only |
| User's actual IRRA best.pth | NOT RUN | File is on Kaggle and unavailable locally |
| Actual ICFG-PEDES official test | NOT RUN | No official dataset mounted; historical R1=63.573%, mAP=38.149% not re-measured |
| CUDA / actual single T4 batch 64 | NOT RUN | 2 CUDA tests skipped; actual full ViT-B/16 memory/time still needs Kaggle smoke |
| Notebook execution top to bottom on Kaggle | NOT RUN | Notebook prepared and structurally tested; Kaggle execution unavailable locally |
| Full one-epoch real-data warm-start / 5-epoch exploratory run | NOT RUN | Commands/notebook ready; only toy one-batch runs were executed here |
| Scientific ablations / final multi-seed study | NOT RUN | No scientific performance or superiority claim |

Reproduce automated checks:

```bash
/tmp/irra-scfp-env/bin/python -m pytest -q
/tmp/irra-scfp-env/bin/python -m experiments.scfp.smoke_test \
  --tiny --device cpu --output_file /tmp/scfp_new_cpu_smoke.json
```

The tiny smoke uses visual width 64, 3 visual blocks, 1 text block, **512-dimensional outputs**, synthetic `[4,3,32,16]` images and a small vocabulary. It is not full ViT-B/16. Observed initial values:

```text
Total loss:                  22.861145
Mean embedding cosine:       1.000000000 (FP32 rounding)
Maximum embedding difference:0.0002080202
Frequency residual norm:     0.01247909
Frequency residual ratio:    0.0002673180
Across-image gate variance:  0.0000271144
Frequency gate gradient norm:0.01653950
Frequency alpha gradient norm:0.01166077
```

Raw report: [reports/cpu_tiny_smoke.json](reports/cpu_tiny_smoke.json). The separate toy DataLoader check used vocabulary 49408 and 512-d outputs with a small width-64 / 3-block visual model; it verified offline checkpoint construction with actual tokenization, not retrieval quality. Toy scientific scores are intentionally not reported as research metrics.

## Exact remaining Kaggle sequence

1. Attach the dataset, actual IRRA checkpoint/config, and current source bundle.
2. Run the full V2 real-batch warm-start smoke. Verify losses/gradients, gate variance, initial embedding cosine/difference/residual and peak T4 memory at batch 64. Lower the batch only if needed, then keep the same value across scientific variants.
3. Run one actual warm-start epoch without validation/test selection. Inspect logs and fixed sample attention/gates. Optionally launch a fresh five-epoch engineering run.
4. Screen all four modes from original CLIP with identical pre-training PID split/seed/augmentation/budget; compare manifests. No full-data IRRA initialization for this phase.
5. Reserve official test for the chosen configuration and report exact metrics. Expand budget/seeds separately.

## Corrections made during validation

An added CPU BF16-autocast test exposed that concatenating FP16 tokens outside the disabled-autocast region could trigger dtype-prioritization errors. Reinsertion now occurs inside that region; the new regression test passes. Saved run configurations now use safe YAML and ignore recomputed runtime fields when used as resume presets. V2 checkpoints with metadata cannot silently warm-start missing newly trained modules, even if every new tensor was removed.

## Git / execution scope

Current branch: `feat/scfp-v2`; base: `cfc23785545ead82dee2749fa0fc0ad948c96430`.
All implementation changes remain uncommitted. No push, merge of LGSP, full real-data training, or experimental checkpoint overwrite occurred. See the source inventory in [README.md](README.md).
