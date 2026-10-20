# LGSP attribution

The new `model/lgsp` implementation is inspired by and adapts the spatial
prompting mechanisms of **Revisiting Pool-based Prompt Learning for Few-shot
Class-incremental Learning**, Yongwei Jiang, Yixiong Zou, Yuhua Li, Ruixuan Li,
ICCV 2025. Paper: https://arxiv.org/abs/2507.09183

Reference code: https://github.com/Jywsuperman/LGSP
Audited revision: `fe2d0619a6a023450a25cf6941de451af8c869d9`.
In particular: `models/base/ViT_Network.py` (`get_prompts`,
`cosine_similarity`, `get_Frequency_mask`, `forward`), `models/prompt.py`,
`models/model_My.py`, and the repository tree were inspected.

This is a compact retrieval adaptation, not the original FSCIL pipeline.
See `experiments/lgsp_tbpr/README.md` for algorithmic differences.
The upstream LGSP notice is retained below. Existing IRRA licensing remains
in the root `LICENSE`.

## LGSP MIT license

MIT License

Copyright (c) 2025 Jiang Yongwei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
