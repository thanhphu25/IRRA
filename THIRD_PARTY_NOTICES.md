# Source and research attribution

The repository retains its existing MIT license and original IRRA/OpenAI CLIP notices. `model/clip_model.py` remains adapted from OpenAI CLIP, MIT, Copyright (c) 2021 OpenAI.

The V2 implementation extends the IRRA baseline at `cfc23785545ead82dee2749fa0fc0ad948c96430` (https://github.com/anosorae/IRRA). `tests/fixtures/irra_build_main.py` is an unmodified snapshot of this repository's baseline `model/build.py` for regression testing.

SCFP research modules were written for this task. Shared semantic prototypes / text-dependent relevance were informed by PLOT (Park et al., ECCV 2024, https://github.com/jicheol93/PLOT). Spatial Fourier band selection was informed by LGSP-Prompt (Jiang et al., ICCV 2025, https://github.com/Jywsuperman/LGSP). No PLOT or LGSP model implementation was copied or vendored. V2 deliberately differs in the mechanisms described in `experiments/scfp/README.md`.

Engineering utilities for metrics, identity holdout and Kaggle preparation were selectively adapted from this user's `feat/lgsp-tbpr` branch. No previous LGSP research architecture was migrated.

EESSO (Image and Vision Computing 2024, https://doi.org/10.1016/j.imavis.2024.104912) is prior frequency-based TBPR work; frequency learning in TBPR is not claimed to originate here. ITSELF (https://trhuuloc.github.io/itself/) is additional conceptual related work. Neither is an implementation dependency.
