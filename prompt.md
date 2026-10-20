# RESEARCH ENGINEERING TASK — IMPLEMENT V2: DUAL-MODAL SEMANTIC & ADAPTIVE FREQUENCY PROMPTING FOR TEXT-BASED PERSON RETRIEVAL

## 0. Role and execution instructions

You are a senior Deep Learning Research Engineer specializing in:

- Text-Based Person Retrieval (TBPR)
- CLIP and Vision-Language Models
- Vision Transformers
- Cross-modal Representation Learning
- Semantic Slot Attention
- Prompt Learning
- Frequency-domain Feature Learning
- PyTorch research implementation

Your task is to **directly implement V2: Dual-modal Semantic & Adaptive Frequency Prompting**, a research architecture built on the existing IRRA repository.

The project aims to investigate whether combining semantic local alignment across image/text modalities with adaptive frequency enhancement of ViT visual features can improve fine-grained Text-Based Person Retrieval.

This is an experimental research implementation intended for future scientific publication.

**You must implement the actual code, not merely propose an architecture.**

Follow a milestone-based workflow:

1. Inspect the current codebase and research references.
2. Preserve the existing IRRA baseline.
3. Implement V2 modules.
4. Integrate V2 with training and inference.
5. Implement ablation configurations.
6. Implement unit tests and smoke tests.
7. Prepare a working Kaggle notebook.
8. Document all modifications and experimental protocols.

Do not stop after writing a plan.

Do not fabricate experimental results.

Do not launch full 60-epoch training locally.

Do not overwrite existing experimental checkpoints.

Do not push commits without explicit permission.

---

# 1. RESEARCH CONTEXT

## 1.1. Problem: Text-Based Person Retrieval

Text-Based Person Retrieval retrieves images of a person from a gallery based on a natural-language description.

Example:

Text query:

"A man wearing a black jacket, blue jeans, white sneakers, and carrying a red backpack."

The retrieval model should return images corresponding to that description.

This is a cross-modal retrieval problem involving two modalities:

- Visual modality: person images.
- Textual modality: natural-language descriptions.

The main challenges include:

1. Cross-modal semantic gap.
2. Fine-grained attribute alignment.
3. Similar clothing and appearances across identities.
4. Occlusion and background noise.
5. Global versus local representation.
6. Information loss when compressing fine-grained features into global embeddings.

Evaluation metrics:

- Rank-1
- Rank-5
- Rank-10
- mAP
- mINP

The primary dataset is ICFG-PEDES.

The goal is to improve retrieval performance over IRRA while maintaining independent image and text encoding at inference.

## 1.2. Existing baseline: IRRA

Paper:

**Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval**

Conference: CVPR 2023

Paper:

https://arxiv.org/abs/2303.12501

Official source:

https://github.com/anosorae/IRRA

User's repository:

https://github.com/thanhphu25/IRRA

IRRA uses:

- CLIP ViT visual encoder.
- CLIP text encoder.
- Shared image-text embedding space.
- Similarity Distribution Matching (SDM).
- Identity classification loss (ID).
- Masked Language Modeling (MLM) with cross-modal interaction.

The existing IRRA architecture should remain the foundation of this project.

Do not rewrite the IRRA training pipeline.

Do not introduce an unrelated backbone.

Do not replace SDM with a new objective.

The new modules must extend IRRA.

### Existing baseline experiment

The user has already trained IRRA successfully on Kaggle.

Configuration:

```text
Dataset: ICFG-PEDES
Backbone: CLIP ViT-B/16
Image resolution: 384 × 128
Patch size: 16
Batch size: 64
Epochs: 60
Optimizer: Adam
Learning rate: 1e-5
Image augmentation: enabled
Losses: SDM + MLM + ID
GPU: Tesla T4
```

Measured official test results:

| Metric | Result |
|---|---:|
| Rank-1 | 63.573% |
| Rank-5 | 80.416% |
| Rank-10 | 85.868% |
| mAP | 38.149% |
| mINP | 7.919% |

The checkpoint has been independently evaluated and loaded with:

```text
Missing keys: []
Unexpected keys: []
```

This is an experimentally verified baseline, not a target guaranteed to be exceeded.

The original checkpoint is available on Kaggle at:

```text
/kaggle/input/datasets/tphu25/irra-icfgpedes-checkpoint/IRRA_runs/ICFG-PEDES/20261008_162321_irra_icfg_paper/best.pth
```

Original configuration:

```text
/kaggle/input/datasets/tphu25/irra-icfgpedes-checkpoint/IRRA_runs/ICFG-PEDES/20261008_162321_irra_icfg_paper/configs.yaml
```

The user also has:

```text
train_features.pt
test_features.pt
```

These feature caches can be used for offline diagnostics but cannot replace re-encoding images through the new feature-frequency module.

### Dataset path

```text
/kaggle/input/datasets/tphu25/icfg-pedes-dataset/ICFG-PEDES/ICFG-PEDES.json
```

Images:

```text
/kaggle/input/datasets/tphu25/icfg-pedes-dataset/ICFG-PEDES/imgs
```

Expected prepared layout:

```text
/kaggle/working/data/
└── ICFG-PEDES/
    ├── ICFG-PEDES.json
    └── imgs/
```

Dataset statistics from the user's baseline:

```text
Training identities: 3102
Training image-caption pairs: 34674

Test identities: 1000
Test image-caption pairs: 19848

Official validation set: unavailable
```

Do not hard-code these statistics into the dataset implementation. Verify actual data when available.

---

# 2. RELATED RESEARCH AND SOURCE CODE

Study the relevant papers before implementation.

The purpose is to transfer useful mechanisms from related research domains into TBPR, rather than inventing unrelated modules.

## 2.1. IRRA — CVPR 2023

Paper:

https://arxiv.org/abs/2303.12501

Source:

https://github.com/anosorae/IRRA

Relevant files:

```text
model/build.py
model/clip_model.py
model/objectives.py
solver/build.py
processor/processor.py
utils/options.py
utils/metrics.py
```

What to preserve:

- CLIP encoder architecture.
- SDM loss.
- MLM branch.
- ID loss.
- Global retrieval embeddings.
- Evaluation protocol.

IRRA's goal is to improve global retrieval representations using fine-grained relationships during training without expensive cross-modal matching at inference.

V2 should preserve this general efficiency advantage.

## 2.2. PLOT — ECCV 2024

Paper:

**PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery**

Official source:

https://github.com/jicheol93/PLOT

Study:

- Part Slot Attention.
- Learnable part/semantic representations.
- Correspondence between image and text features.
- Local feature alignment.
- Attention-based selection.
- Avoiding explicit body-part annotations.

Important:

Do not copy PLOT's entire model.

Use its slot-learning approach as a reference for designing lightweight semantic slot modules compatible with IRRA.

Our proposed V2 is not simply a reproduction of PLOT.

## 2.3. LGSP-Prompt — ICCV 2025

Paper:

**Revisiting Pool-based Prompt Learning for Few-shot Class-incremental Learning**

Paper:

https://arxiv.org/pdf/2507.09183

Official source:

https://github.com/Jywsuperman/LGSP

Relevant code to inspect:

```text
models/base/ViT_Network.py
models/prompt.py
models/model_My.py
```

Study:

- Local Spatial Prompting.
- Convolution-based prompt pool.
- Fourier-domain enhancement.
- Frequency ring masks.
- Learnable frequency selection.
- Spatial fusion.

Important distinction:

The original LGSP applies spatial/frequency enhancement in a different visual learning pipeline.

V2 will use **feature-level frequency enhancement inside CLIP ViT**, not a direct copy of the RGB input transformation.

The goal is to adapt the frequency-learning principle to retrieval.

Do not import FSCIL training loops, sessions, classifier logic, or incremental-learning dependencies.

## 2.4. EESSO — 2024

Reference:

https://doi.org/10.1016/j.imavis.2024.104912

This is an important related work using spatial and frequency information for Text-Based Person Retrieval.

Review its relevant mechanisms when accessible.

V2 must not claim that frequency learning in TBPR is entirely new.

The research question is whether adaptive frequency enhancement of intermediate CLIP features, combined with cross-modal semantic slots, provides additional value over existing methods.

## 2.5. Additional reference

ITSELF — WACV 2026:

https://trhuuloc.github.io/itself/

Study relevant concepts of fine-grained patch/token selection and alignment if the source is accessible.

Treat it as related work, not a mandatory implementation dependency.

---

# 3. PREVIOUS LGSP EXPERIMENT AND LESSONS

The user has already implemented an initial LGSP-inspired frequency module on top of IRRA.

Previous branch:

```text
feat/lgsp-tbpr
```

The previous method applied DFT-based spatial enhancement to the visual input.

The implementation successfully passed:

- Forward.
- Backward.
- Checkpoint warm-start.
- Gradient verification.
- Mixed-precision execution.
- Kaggle T4 smoke test.

Observed diagnostic values:

```text
Initial frequency alpha: 0.001
Initial frequency weights: uniform across 8 rings

After 5 epochs:
Frequency alpha: approximately -0.16
Frequency contribution ratio: approximately 0.025
```

The module learned nonuniform frequency weights, but no reliable improvement in official retrieval performance was established.

The previous experiment used continued fine-tuning from a fully trained IRRA checkpoint.

It also revealed that fine-tuning an already converged backbone with learning rate 1e-5 could destabilize the representation.

There was a validation issue:

The old checkpoint was trained on the full ICFG-PEDES training set.

A holdout subsequently created from that same training set is not an independent validation set for this checkpoint.

Therefore:

**Do not use the previous holdout R1 around 94% as evidence of generalization.**

The verified official test R1 remains 63.573%.

### Lessons for V2

1. Do not simply repeat pixel-level DFT.
2. Move frequency enhancement into intermediate visual features.
3. Use sample-adaptive frequency gates.
4. Connect the visual learning process to local text semantics.
5. Preserve pretrained representations through stable initialization.
6. Use controlled ablation experiments.
7. Implement a scientifically valid validation protocol before drawing conclusions.

---

# 4. V2 RESEARCH OBJECTIVE

## Temporary method name

**V2 — Dual-modal Semantic & Adaptive Frequency Prompting**

This is a working research name.

Do not claim novelty merely because this combination has a different implementation.

The method must be tested against related approaches.

## Main hypothesis

Combining:

1. Adaptive frequency enhancement of intermediate CLIP visual features.
2. Semantic local representations from image patches.
3. Semantic local representations from text tokens.
4. Cross-modal alignment between corresponding semantic slots.

may improve fine-grained text-based person retrieval.

The proposed model should retain IRRA's global retrieval structure.

---

# 5. V2 ARCHITECTURE

The architecture consists of two parallel encoding pipelines with supplementary cross-modal learning.

## Image pipeline

```text
Person Image
    |
    v
CLIP ViT Patch Embedding
    |
    v
ViT Blocks 1–6
    |
    v
Adaptive Feature Frequency Module
    |
    v
ViT Blocks 7–12
    |
    +----------------------+
    |                      |
    v                      v
Global CLS Feature    Patch Features
    |                      |
    v                      v
Global Image Embed    Visual Semantic Slots
                           |
                           v
                      K Local Features
```

The frequency module is inserted after an intermediate ViT block.

Use block 6/12 as the initial experiment.

Support changing the insertion block via configuration.

Do not assume this is the optimal position.

## Text pipeline

```text
Text Description
    |
    v
CLIP Text Encoder
    |
    +-----------------------+
    |                       |
    v                       v
Global EOT Feature       Token Features
    |                       |
    v                       v
Global Text Embed        Text Semantic Slots
                            |
                            v
                       K Local Features
```

No DFT is applied to textual tokens.

Text is processed semantically, not as a spatial 2D frequency signal.

## Training alignment

```text
Global Image Embedding
          |
          +------ IRRA SDM / ID / MLM
          |
Global Text Embedding


Visual Semantic Slots
          |
          +------ Local Cross-modal Alignment
          |
Text Semantic Slots


Visual + Text Slot Embeddings
          |
          +------ Slot Diversity Regularization
```

Training objective:

```text
L_V2 = L_IRRA + lambda_slot * L_slot
              + lambda_div * L_div
```

Where:

```text
L_IRRA = original weighted SDM + ID + MLM objectives
```

Do not change the existing loss weights or the behavior of the original IRRA objectives.

### Inference

The default inference must still be:

```text
Person Image → Visual Encoder → Global Image Embedding

Text Query → Text Encoder → Global Text Embedding

Similarity → Ranking
```

Semantic slots provide auxiliary supervision during training.

Do not require pairwise local reranking at inference.

Do not introduce cross-attention between every gallery image and text query at inference.

Gallery image embeddings must remain precomputable.

---

# 6. MODULE A — ADAPTIVE FEATURE-LEVEL FREQUENCY LEARNING

This is the first major research component.

## 6.1. Input representation

CLIP ViT-B/16 receives images at:

```text
[B, 3, 384, 128]
```

Patch grid:

```text
24 × 8 = 192 spatial patches
```

ViT hidden width:

```text
768
```

At the selected intermediate transformer block:

```text
[B, 193, 768]
```

The 193 tokens include:

```text
1 CLS token + 192 patch tokens
```

The exact tensor layout at the selected hook location may differ depending on the implementation.

Inspect the source.

Do not assume batch-first tensors internally.

## 6.2. Spatial transformation

Separate CLS token from patch tokens.

Reshape:

```text
Patch tokens:
[B, 192, 768]

Spatial feature grid:
[B, 768, 24, 8]
```

For a generic input resolution:

```text
H_grid = H / patch_size
W_grid = W / patch_size
```

Use the actual patch embedding output and stride.

Do not hard-code 24×8.

The module must support rectangular grids.

## 6.3. Fourier transformation

Let:

```text
V ∈ R[B,C,Hg,Wg]
```

Compute a 2D FFT over the spatial dimensions:

```python
F = torch.fft.fft2(
    V.float(),
    dim=(-2, -1),
    norm="ortho"
)
```

The Fourier transform must not operate over the feature-channel dimension.

Construct frequency masks in the 2D spatial-frequency domain.

Use normalized radial frequency coordinates suitable for nonsquare grids.

Initially implement:

```text
R = 4 frequency bands
```

The number of bands must be configurable.

Use masks that consistently cover the frequency bins without accidental overlap or omission.

Document whether masks use hard boundaries or differentiable soft boundaries.

## 6.4. Image-adaptive frequency gating

Unlike the previous LGSP implementation, frequency weights should depend on the input image.

Use a lightweight gating network:

```text
Feature Grid
    |
    v
Global Spatial Pooling
    |
    v
Small MLP
    |
    v
R Frequency Logits
    |
    v
Frequency Weights
```

For example:

```text
pooled = GAP(V)

logits = MLP(pooled)

weights = softmax(logits / temperature)
```

Thus:

```text
weights = weights(image)
```

rather than one globally shared weight vector for all images.

The gating network should be small enough for a single T4 GPU.

A reasonable initial hidden dimension is 64 or 128.

Document the choice and expose it through configuration.

## 6.5. Frequency enhancement

Let `M_r` denote frequency band masks.

Use a near-identity frequency response such as:

```text
G(I) = 1 + alpha_f *
       Σ_r [R * weight_r(I) - 1] * M_r
```

Then:

```text
F_enhanced = G(I) * F

V_enhanced = IFFT2(F_enhanced)
```

This formulation is suggested for the V2 adaptation, not claimed to be the exact LGSP formula.

Alternative mathematically equivalent or more stable parameterizations are acceptable if justified.

Important:

- The initial gate should be close to identity.
- Frequency weights must remain learnable.
- A sample-specific frequency response must be produced.
- All gradients should propagate correctly.
- Do not add the original feature map twice accidentally.
- Do not apply `abs()` to IFFT output without a clearly justified reason.
- Handle conjugate symmetry appropriately to produce real-valued outputs.
- Verify that imaginary numerical residuals remain negligible.
- Avoid unbounded negative or extremely large frequency gains.

Use FP32 for FFT/IFFT even when the surrounding ViT uses FP16.

Cast back to the backbone-compatible dtype before reinserting tokens.

## 6.6. Reintegration into ViT

Transform the enhanced grid back:

```text
[B,768,24,8]
    ↓
[B,192,768]
```

Reattach the untouched CLS token at the same sequence position.

Pass the complete sequence through the remaining ViT blocks.

The frequency module must affect both:

- Global CLS representation.
- Final visual token representations used by MLM/local alignment.

Do not perform frequency transformation separately inside train and test functions.

Prefer a single visual encoding pathway with an optional intermediate adapter.

## 6.7. Required frequency diagnostics

Track:

```text
frequency_alpha
frequency_weights_mean
frequency_weights_std
frequency_weights_entropy
frequency_gate_variance_across_images
frequency_residual_norm
frequency_residual_ratio
frequency_gradient_norm
```

Also support saving the frequency weights for a small fixed sample set.

An important research criterion is that frequency weights actually depend on image content.

Check this explicitly.

If all images receive nearly identical weights, the adaptive gate may have collapsed to a global filter.

Prepare interfaces for V3 to access:

```python
frequency_logits
frequency_weights
frequency_band_features
```

Do not implement V3 teacher/guidance yet.

---

# 7. MODULE B — VISUAL SEMANTIC SLOTS

This is the second major component.

## 7.1. Purpose

Learn K local representations from visual patch tokens.

The goal is to capture discriminative fine-grained information useful for text-based matching.

Examples of desired semantic information:

- Upper-body clothing.
- Lower-body clothing.
- Footwear.
- Accessories.
- Color and texture.

These are desired behaviors, not predefined semantic labels.

The model must learn local features without part-level annotations.

## 7.2. Input/output

Use final projected CLIP visual patch tokens, or a properly projected equivalent.

Expected:

```text
Input:
[B,192,512]

Output:
[B,K,512]
```

Default:

```text
K = 4
```

Make K configurable.

## 7.3. Slot attention

Use learnable semantic queries.

Let:

```text
Q = learnable slot queries
V = visual patch features
```

Compute:

```text
Visual Slots = CrossAttention(Q, V, V)
```

A multi-head attention module is sufficient for the first version.

Recommended:

```text
4 attention heads
K = 4 semantic queries
```

Support:

- LayerNorm.
- Lightweight feed-forward projection.
- Normalized output embeddings.
- Attention weights for diagnostics.

Do not import the entire PLOT model if a small self-contained implementation is sufficient.

Do not introduce additional segmentation networks.

Do not use manually labeled body parts.

---

# 8. MODULE C — TEXT SEMANTIC SLOTS

The third major component is local semantic representation learning for text.

## 8.1. Input

IRRA's CLIP text encoder already provides token-level features.

Expected shape:

```text
[B,77,512]
```

Verify this in the actual implementation.

## 8.2. Semantic queries

Implement a text-side semantic slot module:

```text
Text Tokens
    |
    v
CrossAttention(Semantic Queries, Text Tokens)
    |
    v
K Text Slots
```

Output:

```text
[B,K,512]
```

## 8.3. Shared semantic space

Visual slots and text slots should represent compatible semantic concepts.

Suggested implementation:

- Shared latent semantic slot prototypes.
- Modality-specific query projections.
- Modality-specific key/value projections.
- Shared final embedding dimension.

Example:

```text
Shared Semantic Queries
          |
          +---- Visual Query Projection
          |
          +---- Text Query Projection
```

The exact mechanism may be adapted after inspecting PLOT.

Do not assume that simply assigning the same slot index on both modalities guarantees semantic correspondence.

The alignment loss and diagnostics must help verify correspondence.

## 8.4. Token masking

Correctly mask:

- Padding tokens.
- Non-content special tokens where appropriate.
- Positions beyond EOT.

Use the repository's actual CLIP tokenizer conventions.

In CLIP, EOT is typically identified by the highest token ID within the sequence, but inspect the tokenizer rather than blindly assuming universal behavior.

Avoid attending to padding tokens.

Handle very short and empty-content descriptions without producing NaNs.

## 8.5. Text slot relevance

Implement lightweight slot relevance estimation:

```text
relevance = sigmoid(MLP(text_slots))
```

or another stable mechanism.

Its purpose is to reduce the effect of text slots that represent attributes not described in a particular sentence.

However, a relevance network is not automatically semantically supervised.

Avoid claiming it learns explicit attribute presence without empirical validation.

Prevent trivial collapse in which all relevance weights approach zero.

Record relevance statistics during training.

---

# 9. MODULE D — LOCAL CROSS-MODAL ALIGNMENT

## 9.1. Objective

Align local visual and textual semantic representations while preserving the original global IRRA objective.

Let:

```text
Visual slots:
V_slots ∈ R[B,K,512]

Text slots:
T_slots ∈ R[B,K,512]
```

A simple corresponding-slot similarity is:

```text
S_local(i,j) =
    Σ_k relevance(j,k) *
    cosine(V_slots[i,k], T_slots[j,k])
```

Normalize relevance weights so score magnitude is consistent across different queries.

This is the initial alignment formulation.

If PLOT provides a better established slot correspondence approach that is compatible with IRRA, compare it before finalizing the implementation.

Do not implement a complicated optimal-transport matching pipeline unless there is a clear need.

## 9.2. Batch-level local similarity

Construct:

```text
S_local ∈ R[B,B]
```

All image-text pairs in a batch should be compared.

The training batch contains person identities:

```text
batch['pids']
```

Use identity matching to determine positives.

Very important:

ICFG-PEDES can contain multiple images or captions associated with one person.

Do not assume only diagonal image-text pairs are positive.

For query i and gallery item j:

```text
positive(i,j) = (pid_i == pid_j)
```

Use a multi-positive matching objective compatible with IRRA's existing SDM philosophy.

Suggested approach:

- Normalize row-wise multi-positive targets.
- Apply similarity distribution matching or a stable multi-positive contrastive objective.
- Calculate both image-to-text and text-to-image directions.
- Handle repeated identities correctly.
- Handle small batches safely.

If possible, reuse the existing SDM formulation with local similarity scores rather than inventing a fundamentally different objective.

## 9.3. Slot collapse prevention

A known risk is that all K slots become nearly identical.

Implement lightweight diversity regularization.

Possible mechanisms:

- Penalize off-diagonal cosine similarity between normalized slot embeddings.
- Encourage attention diversity across slots.
- Monitor slot usage entropy.

Choose one simple regularizer for the initial experiment.

Avoid very strong diversity penalties that force semantically similar attributes to separate artificially.

The diversity loss must be optional and independently switchable.

Recommended initial objective:

```text
L_total =
    L_IRRA
    + lambda_slot * L_slot
    + lambda_div * L_div
```

Initial hyperparameters to expose:

```text
lambda_slot = 0.1
lambda_div = 0.01
```

These are exploratory defaults, not established optimal values.

Support:

```text
lambda_slot = 0
lambda_div = 0
```

for ablations.

Do not introduce another loss unless necessary for correct implementation.

---

# 10. FULL V2 TRAINING OBJECTIVE

The original IRRA objectives remain unchanged.

Total loss:

```text
L_V2 =
    L_SDM
    + L_ID
    + L_MLM
    + lambda_slot * L_slot
    + lambda_div * L_div
```

Retain the original IRRA weighting of ID/MLM losses.

Do not double-apply existing loss weights.

Required logging:

```text
sdm_loss
id_loss
mlm_loss
slot_loss
diversity_loss
total_loss
```

Provide optional gradient norms for:

```text
Visual Semantic Slots
Text Semantic Slots
Frequency Gate
Frequency Residual Scale
CLIP Vision Backbone
CLIP Text Backbone
```

Keep local slot matching in the training path.

By default, inference should still use global embeddings.

If the semantic slot modules are disabled:

```text
IRRA output must match the original baseline.
```

---

# 11. COMPATIBILITY WITH IRRA

Before making changes, inspect:

```text
model/build.py
model/clip_model.py
model/objectives.py
model/__init__.py

solver/build.py

processor/processor.py

utils/options.py
utils/metrics.py
utils/checkpoint.py

datasets/build.py

train.py
test.py
```

Some files may differ in the user's fork. Use actual source paths rather than creating duplicate modules unnecessarily.

## 11.1. Training/inference compatibility

In upstream IRRA, the training model calls:

```python
image_feats, text_feats = self.base_model(
    images,
    caption_ids
)
```

Image inference calls:

```python
self.base_model.encode_image(image)
```

Text inference calls:

```python
self.base_model.encode_text(text)
```

Ensure the frequency-enhanced visual encoding is the same for training and inference.

Do not implement separate inconsistent visual transformations.

The existing MLM module must consume visual tokens corresponding to the frequency-enhanced input.

## 11.2. Hidden features

The baseline CLIP implementation already computes token-level features.

Prefer reusing them.

Avoid running CLIP Vision twice per batch.

Avoid running CLIP Text twice when unnecessary.

If extracting intermediate features requires changing `clip_model.py`, keep the patch minimal and preserve pretrained parameter names.

Avoid unnecessary duplication of Transformer blocks.

## 11.3. Original losses

Do not alter:

```text
SDM loss
ID loss
MLM loss
```

unless fixing a verified bug.

Any unrelated modification must be separately documented and regression-tested.

## 11.4. Optimizer

Ensure all new parameters are correctly registered.

Check optimizer groups carefully because IRRA has separate learning-rate behavior for different parameter groups.

Each trainable parameter should appear in exactly one optimizer group.

No new module should accidentally remain frozen.

Expose a separate learning rate for new SCFP modules.

---

# 12. ABLATION DESIGN

There are only two intended research versions in the broader project:

```text
V2 — Semantic Slots + Adaptive Feature Frequency
V3 — V2 + Semantic-guided Frequency Alignment
```

This task implements V2 only.

However, V2 needs internal ablations.

Support four modes:

| Mode | Semantic Slots | Feature DFT |
|---|---|---|
| baseline | OFF | OFF |
| slots | ON | OFF |
| frequency | OFF | ON |
| v2 | ON | ON |

Suggested CLI:

```bash
--scfp_mode baseline
--scfp_mode slots
--scfp_mode frequency
--scfp_mode v2
```

All modes must use the same backbone and existing IRRA objectives.

For `slots` and `v2`, enable local alignment.

For `baseline` and `frequency`, do not compute slot losses.

The ablation system must not require manual source editing.

The default mode must preserve IRRA baseline behavior.

---

# 13. CONFIGURATION PARAMETERS

Add flags following the repository's existing conventions.

Suggested:

```text
--scfp_mode
--scfp_num_slots
--scfp_slot_heads
--scfp_slot_loss_weight
--scfp_diversity_weight
--scfp_slot_dropout

--scfp_frequency_layer
--scfp_frequency_num_bands
--scfp_frequency_hidden_dim
--scfp_frequency_temperature
--scfp_frequency_alpha_init
--scfp_frequency_variant

--scfp_lr
--init_checkpoint
--freeze_backbone

--scfp_diagnostics
```

Recommended exploratory defaults:

```text
Mode: v2

Number of slots: 4
Slot attention heads: 4

Frequency insertion layer: 6
Frequency bands: 4
Frequency gate hidden dimension: 64

Slot loss weight: 0.1
Diversity loss weight: 0.01

Frequency alpha initialization: 0.01
```

All parameters must be saved in `configs.yaml`.

Checkpoint reload must reconstruct the correct model architecture without silently selecting baseline mode.

Avoid letting old `lgsp_mode` settings override `scfp_mode`.

---

## 14. REPOSITORY AND BRANCH STRATEGY — MANDATORY

Use `main` as the ONLY base branch for the new V2 implementation.

Repository:
https://github.com/thanhphu25/IRRA

Create:
`feat/scfp-v2`

**Do not create V2 from `feat/lgsp-tbpr`. Do not merge the LGSP branch into V2.**

The existing `feat/lgsp-tbpr` branch is a previous experimental implementation based on pixel-level frequency prompting. V2 is a different architecture involving intermediate CLIP feature-level DFT and dual-modal semantic slots.

### Required procedure

1. Inspect Git status and preserve uncommitted changes.
2. Fetch the latest remote branches.
3. Switch to `main` and create `feat/scfp-v2`.
4. Inspect the LGSP branch only to identify reusable, stable engineering infrastructure.
5. Selectively migrate and adapt useful utilities:
   - Checkpoint warm-start
   - Chunked retrieval evaluation
   - Smoke-test infrastructure
   - Training diagnostics and logging
   - Kaggle dataset preparation
   - Checkpoint saving/resuming
6. Reimplement the V2 research architecture independently on top of IRRA.

### Strict exclusions

Do NOT migrate:
- Old pixel-level LGSP frequency modules.
- RGB spatial prompt fusion.
- Old local convolutional prompt pool.
- LGSP-specific model wrappers or training logic.
- Old architecture-specific checkpoint/configuration behavior.

Never cherry-pick a commit that mixes infrastructure with LGSP model changes without separating the relevant modifications.

Ensure V2 uses only the newly implemented SCFP architecture.

Preserve backward compatibility with the original IRRA `best.pth`.

Before implementation, report:
- Base commit from `main`
- Reusable LGSP infrastructure identified
- Infrastructure actually migrated
- Old architecture components explicitly excluded

The final repository must clearly separate IRRA baseline, previous LGSP experiments, and the new SCFP V2 implementation.


---

# 15. SUGGESTED SOURCE STRUCTURE

Use a clean modular layout such as:

```text
IRRA/
├── model/
│   ├── build.py
│   ├── clip_model.py
│   ├── objectives.py
│   └── scfp/
│       ├── __init__.py
│       ├── feature_frequency.py
│       ├── frequency_gate.py
│       ├── semantic_slots.py
│       ├── local_alignment.py
│       ├── diversity.py
│       └── model_adapter.py
│
├── solver/
│   └── build.py
│
├── utils/
│   ├── options.py
│   ├── checkpoint.py
│   └── metrics.py
│
├── experiments/
│   └── scfp/
│       ├── README.md
│       ├── smoke_test.py
│       ├── diagnostics.py
│       └── configs/
│
├── tests/
│   ├── test_feature_frequency.py
│   ├── test_semantic_slots.py
│   ├── test_local_alignment.py
│   ├── test_model_integration.py
│   └── test_checkpoint_compatibility.py
│
├── notebooks/
│   └── irra_scfp_v2_icfg_kaggle.ipynb
│
├── train.py
└── test.py
```

This layout is a suggestion.

Adapt it to the existing repository.

Avoid excessive boilerplate and unnecessary framework abstractions.

---

# 16. CHECKPOINT STRATEGY

The user has an existing pretrained IRRA checkpoint:

```text
best.pth
```

Support:

```bash
--init_checkpoint /path/to/best.pth
```

Warm-start behavior:

1. Load compatible pretrained IRRA weights.
2. Initialize new V2 parameters separately.
3. Report missing/unexpected keys.
4. Verify there are no incompatible existing IRRA tensors.
5. Create a new optimizer.
6. Create a new scheduler.
7. Begin a new experiment from epoch 1.

Do not restore old optimizer/scheduler state during warm-start.

Distinguish:

```text
init_checkpoint
resume_checkpoint
```

New V2 checkpoints must include:

- IRRA backbone.
- Original IRRA heads.
- Frequency module.
- Visual semantic slots.
- Text semantic slots.
- Local alignment parameters.
- Config metadata.
- Optimizer/scheduler when appropriate for resuming.

Loading a trained V2 checkpoint must not leave any new module randomly initialized without warning.

Support old IRRA checkpoints while rejecting unrelated missing keys.

## Near-identity initialization

When introducing frequency enhancement, initially preserve pretrained image representations as much as practical.

Test:

```text
Cosine similarity between original IRRA embeddings and newly initialized frequency-enhanced embeddings.
```

Report:

```text
mean cosine similarity
maximum embedding difference
frequency residual norm
```

Avoid exact-zero initialization that accidentally prevents new module gradients from flowing.

---

# 17. TRAINING STRATEGY

Support two distinct protocols.

## 17.1. Protocol A — Engineering warm-start

Purpose:

- Smoke testing.
- Gradient verification.
- Numerical stability.
- GPU feasibility.
- Checkpoint compatibility.

Initialization:

```text
Existing IRRA best.pth
```

Recommended initial setup:

```text
Epochs: 1, then 5
Batch size: 64
Backbone LR: approximately 1e-6
New module LR: approximately 1e-4
```

These are starting points for testing, not fixed optimal hyperparameters.

First run:

```text
baseline
slots
frequency
v2
```

with small smoke-test budgets.

Do not automatically train every ablation for 5 epochs.

Initially verify the full V2 forward/backward and one epoch.

The old checkpoint was trained on all ICFG-PEDES training identities.

Therefore, a holdout split constructed afterward cannot serve as an independent validation set for this warm-start model.

Treat this phase strictly as an engineering diagnostic.

Do not use the holdout result to declare a research improvement.

## 17.2. Protocol B — Scientific comparison

Purpose:

- Actual ablation.
- Hyperparameter selection.
- Retrieval improvement verification.

Initialization:

```text
Original pretrained CLIP ViT-B/16
```

Create a fixed identity-disjoint train/validation partition before training.

All experimental variants must use:

- The same training identities.
- The same validation identities.
- The same seed.
- The same image augmentation.
- The same batch size.
- The same training budget.
- Comparable optimization schedules.

Suggested screening:

```text
10–15 epochs
```

for:

```text
baseline
slots
frequency
v2
```

If results are promising, extend training to:

```text
30 epochs
```

Then perform the final study:

```text
60 epochs
3 seeds
```

Do not treat 10–15 epochs as conclusive if models have not converged.

Use official ICFG-PEDES test only after deciding the configuration and protocol.

Keep validation selection separate from final test reporting.

---

# 18. KAGGLE REQUIREMENTS

The final implementation must run on Kaggle.

Expected initial environment:

```text
Single Tesla T4 GPU
Approximately 16 GB VRAM
CUDA-enabled PyTorch
ICFG-PEDES mounted through Kaggle Input
Existing IRRA checkpoint mounted through Kaggle Input
```

The previous LGSP experiment used:

```text
PyTorch: 2.11.0+cu128
CUDA: 12.8
Python: 3.13.15
GPU: Tesla T4
```

Do not assume exactly these versions are always available. Verify the runtime dynamically.

Do not install the old LGSP PyTorch 1.12 environment into the Kaggle notebook.

Use compatible modern PyTorch operations.

## Kaggle notebook

Create:

```text
notebooks/irra_scfp_v2_icfg_kaggle.ipynb
```

It must be executable from top to bottom.

Provide a configuration cell:

```python
EXPERIMENT = "v2"

TRAINING_PROTOCOL = "warmstart"

EPOCHS = 1
BATCH_SIZE = 64
SEED = 42

INIT_CHECKPOINT = None

BACKBONE_LR = 1e-6
SCFP_LR = 1e-4

RUN_UNIT_TESTS = True
RUN_SMOKE_TEST = True
RUN_BASELINE_EVALUATION = False
RUN_FINAL_TEST = False
```

Notebook requirements:

1. Configure single-GPU usage.
2. Clone the correct research branch or load a source ZIP.
3. Install dependencies.
4. Prepare ICFG-PEDES paths.
5. Find checkpoint and original configuration.
6. Load selected experiment mode.
7. Run unit tests.
8. Run model smoke test.
9. Execute selected training mode.
10. Save checkpoints/logs.
11. Evaluate when explicitly requested.
12. Print paths to saved outputs.

Do not use fragile notebook cells that overwrite entire source files.

Make changes in the Git repository instead.

Avoid hard-coded Kaggle Input dataset slugs where auto-discovery can safely identify a unique source.

If multiple candidates exist, require explicit configuration rather than silently picking one.

## Output organization

Suggested:

```text
/kaggle/working/SCFP_V2_runs/
├── A0_baseline/
├── A1_slots/
├── A2_frequency/
└── V2_full/
```

Each experiment should save:

```text
configs.yaml
environment.json
train_log.txt

training_stats.jsonl
validation_metrics.jsonl

best.pth
last.pth

diagnostics.json
metrics.json
```

Save checkpoint and experiment provenance, including the Git commit when available.

Do not overwrite existing runs.

Support resume from `last.pth`.

---

# 19. EVALUATION

Preserve the existing exact chunked evaluation mechanism from the user's prior LGSP branch.

Training-time evaluation:

```text
Rank-1
Rank-5
Rank-10
```

Final evaluation:

```text
Rank-1
Rank-5
Rank-10
mAP
mINP
```

Ensure:

- Exact retrieval ranking.
- Correct identity matching.
- No missing positives silently counted as regular queries.
- Correct metric scaling.
- Chunked metrics match full-matrix metrics on small synthetic data.
- Evaluation model mode matches checkpoint metadata.
- The best checkpoint selection uses an appropriate validation split.

Do not replace exact mAP with top-k-only approximations.

Default retrieval remains based on global image/text embeddings.

No cross-modal reranking should be required.

---

# 20. MANDATORY TESTS

Implement automated tests.

## 20.1. Frequency module tests

Test:

```text
Input [2,193,768]
Grid 24×8
Output [2,193,768]
```

Also test another valid rectangular patch grid.

Validate:

- Shape.
- CLS preservation at the injection operation.
- FP32 FFT behavior.
- FP16-compatible surrounding pipeline.
- Finite outputs.
- Backpropagation.
- Frequency logits gradients.
- Frequency gate gradients.
- Near-identity initialization.
- Sample-dependent weights.
- FFT/IFFT numerical reconstruction.
- Safe behavior under zero or very small residual scales.

## 20.2. Semantic slot tests

Test:

```text
Visual input [2,192,512]
Text input [2,77,512]

Visual slots [2,4,512]
Text slots [2,4,512]
```

Validate:

- Attention masking.
- Padding exclusion.
- Output normalization.
- Gradient propagation.
- Slot relevance.
- Stable handling of short descriptions.
- No NaNs.
- Different slots are not trivially identical at initialization.

## 20.3. Local loss tests

Use synthetic batches containing repeated identity labels.

Example:

```text
PIDs = [1,1,2,3]
```

Ensure two samples of identity 1 are treated as positives.

Test:

- Multi-positive targets.
- Bidirectional local loss.
- Finite gradients.
- Correct matching shapes.
- All-positive and other degenerate small-batch edge cases.

## 20.4. Model integration

Verify all four modes:

```text
baseline
slots
frequency
v2
```

For each:

- Forward pass.
- Backward pass.
- Output embedding dimension = 512.
- Original IRRA loss keys remain valid.
- Correct additional losses.
- Proper optimizer registration.
- Correct inference.
- Checkpoint save/reload.

## 20.5. Baseline regression

When:

```text
scfp_mode = baseline
```

the model must reproduce original IRRA behavior within normal numerical tolerance.

If the original checkpoint and full dataset are available, compare official retrieval metrics.

Expected historical baseline:

```text
R1: 63.573%
mAP: 38.149%
```

Do not invent a test PASS if the full evaluation has not actually been executed.

## 20.6. Kaggle smoke test

Run a single batch when possible.

Record:

```text
Model mode
Backbone
Image shape
Text token shape
Visual slot shape
Text slot shape
Frequency feature shape

SDM loss
MLM loss
ID loss
Slot loss
Diversity loss
Total loss

Gradient norms
Trainable parameters
Peak GPU memory
Forward time
Backward time
```

If GPU, dataset or checkpoint is unavailable locally, run CPU tests and document the exact remaining Kaggle checks.

---

# 21. RESEARCH DIAGNOSTICS

Implement optional diagnostics to support future paper analysis.

## Semantic slots

Track:

```text
Visual slot similarity matrix
Text slot similarity matrix
Slot attention entropy
Slot diversity loss
Text slot relevance
Slot attention maps
```

Optionally save a few visualizations of attention over the person image.

Do not claim a slot corresponds to a body part unless the attention visualization supports that interpretation.

## Frequency learning

Track:

```text
Per-image frequency weights
Mean frequency weights
Frequency-weight variance
Frequency entropy
Residual scale
Residual contribution ratio
Gradient norms
```

Plot learned frequency weighting for selected images when requested.

## Cross-modal local alignment

Track:

```text
Positive-pair local similarity
Negative-pair local similarity
Positive/negative similarity margin
Slot matching statistics
```

These diagnostics help determine whether the modules truly learn meaningful features rather than merely adding parameters.

---

# 22. IMPLEMENTATION PHASES

Complete the task milestone by milestone.

## P0 — Repository audit and setup

Tasks:

- Inspect current branch and code.
- Inspect LGSP/PLOT references.
- Identify reusable infrastructure.
- Create research branch.
- Preserve baseline behavior.
- Verify checkpoint loading logic.
- Establish training/evaluation protocol.

Deliverable:

```text
Clean V2 development branch.
Baseline regression tests.
Architecture integration plan.
```

Do not spend excessive time writing planning documents. Proceed to implementation.

## P1 — Feature Frequency Module

Tasks:

- Implement intermediate ViT feature extraction/injection.
- Implement FFT frequency bands.
- Implement image-adaptive gates.
- Implement near-identity residual.
- Add configuration.
- Add diagnostics.
- Run unit tests.

Deliverable:

```text
IRRA + Adaptive Feature Frequency
```

This is the `frequency` ablation mode.

Complete and validate this module before adding semantic slots.

## P2 — Dual-modal Semantic Slots

Tasks:

- Implement Visual Semantic Slots.
- Implement Text Semantic Slots.
- Implement shared semantic query mechanism.
- Implement correct text masks.
- Implement slot relevance.
- Implement local multi-positive alignment.
- Implement diversity regularization.
- Run tests.

Deliverable:

```text
IRRA + Dual-modal Semantic Slots
```

This is the `slots` ablation mode.

## P3 — Full V2 Integration

Tasks:

- Combine frequency learning and semantic slots.
- Integrate existing IRRA losses.
- Ensure image/text global embeddings remain compatible.
- Ensure training/inference consistency.
- Support all four ablation modes.
- Verify optimizer parameter groups.
- Verify checkpoint compatibility.
- Run integration tests.

Deliverable:

```text
Full V2 architecture.
```

## P4 — Kaggle execution pipeline

Tasks:

- Build notebook.
- Implement dataset preparation.
- Implement checkpoint discovery.
- Implement smoke test.
- Implement single-epoch training.
- Implement experiment logging.
- Implement checkpoint saving/resume.
- Provide executable commands.

Deliverable:

```text
Reproducible Kaggle notebook.
```

Do not run 60 epochs automatically.

## P5 — Documentation and handoff

Tasks:

- Write research README.
- Document architecture.
- Document tensor dimensions.
- Document transferred ideas.
- Document deviations from PLOT/LGSP.
- Document tests.
- Document Kaggle instructions.
- Document ablation protocol.

Deliverable:

```text
V2 implementation ready for Kaggle experiments.
```

---

# 23. DEFINITION OF DONE

The implementation is complete only when:

- [ ] Research branch exists.
- [ ] Original IRRA baseline is preserved.
- [ ] Feature-level FFT module is implemented.
- [ ] Adaptive frequency gate is implemented.
- [ ] Visual Semantic Slots are implemented.
- [ ] Text Semantic Slots are implemented.
- [ ] Cross-modal local alignment is implemented.
- [ ] Diversity regularization is implemented.
- [ ] Original SDM/MLM/ID remain functional.
- [ ] All four ablation modes work.
- [ ] Global image/text inference is independent.
- [ ] Existing IRRA checkpoint warm-start works.
- [ ] New V2 checkpoints save/reload correctly.
- [ ] New parameters receive gradients.
- [ ] Unit tests are implemented.
- [ ] Integration tests are implemented.
- [ ] Kaggle notebook is prepared.
- [ ] Logs and diagnostics are saved.
- [ ] Research README is complete.
- [ ] No fabricated test results.
- [ ] No automatic 60-epoch training.

If any requirement is blocked, clearly state why and which part is unverified.

---

# 24. FUTURE V3 COMPATIBILITY

The broader research plan includes:

**V3 — Semantic-guided Frequency–Spatial Alignment**

V3 will investigate whether semantic information from text can guide the frequency selection process during training.

Possible future design:

```text
Semantic Teacher:
Image Slots + Text Slots → Frequency Guidance Distribution

Image-only Student:
Image Features → Adaptive Frequency Distribution

Teacher/Student Alignment:
Consistency or Distillation Loss
```

However:

**DO NOT IMPLEMENT V3 IN THIS TASK.**

Only ensure V2 exposes:

```python
frequency_logits
frequency_weights
frequency_band_features

visual_slots
text_slots
slot_relevance
```

through an appropriate optional diagnostics/feature interface.

Do not add V3 teacher modules, cross-modal frequency guidance losses, or distillation yet.

Keep the V2 implementation self-contained.

---

# 25. FINAL RESPONSE REQUIREMENTS

After completing implementation, report in Vietnamese.

Use this structure:

### A. Architecture implemented

Explain the actual V2 pipeline:

```text
CLIP Intermediate Features
→ Adaptive Frequency
→ Final Visual Features
→ Semantic Slots
→ Cross-modal Local Alignment
```

Also explain the text pipeline.

### B. Source modifications

List each modified and newly created file and its purpose.

### C. Paper-to-code mapping

Explain which implementation ideas came from:

- IRRA
- PLOT
- LGSP

Clearly identify original code, adapted code and newly written experimental modules.

### D. Tests executed

Use:

```text
Test name
PASS / FAIL / NOT RUN
Observed result
```

### E. Git information

Report:

```text
Current branch
Base commit
Modified files
Uncommitted changes
```

Do not push without permission.

### F. Kaggle execution

Provide exact commands to:

1. Prepare dataset.
2. Attach best.pth.
3. Run baseline regression test.
4. Run V2 smoke test.
5. Train V2 for 1 epoch.
6. Run V2 for 5 exploratory epochs.
7. Run the four ablations under a valid evaluation protocol.
8. Resume from last.pth.
9. Export logs, metrics and checkpoints.

### G. Research limitations

Explain any risks such as:

- Frequency gating collapse.
- Semantic slot collapse.
- Slot index mismatch across modalities.
- Additional GPU memory.
- Backbone forgetting during warm-start.
- Validation leakage.
- Limits of the current experiment for scientific novelty.

Do not claim the proposed method is already novel or superior to prior work.

---

# 26. FINAL PRIORITY

The most important requirement is:

**Produce a correct, modular, checkpoint-compatible, Kaggle-ready implementation of V2 that can evaluate the contribution of semantic slots and adaptive feature-level DFT independently and jointly.**

The first experiment after coding should be:

```text
V2 full-model smoke test
```

using the user's already trained IRRA checkpoint.

Then:

```text
1-epoch V2 warm-start training
```

Then:

```text
Short ablation screening under a controlled protocol
```

Full 60-epoch training will be decided separately after the implementation has been tested.

Begin by inspecting the repository, source references and Git state. Then implement the work milestone by milestone, run applicable tests, and deliver the working code.
