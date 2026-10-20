# Báo cáo xác minh — 2026-10-20

## Môi trường và phạm vi

- Windows, Python 3.12.5 trong `.venv` riêng của workspace.
- PyTorch `2.14.1+cpu`, torchvision `0.29.1+cpu`; CUDA không có.
- Git base `cfc23785545ead82dee2749fa0fc0ad948c96430`, nhánh `feat/lgsp-tbpr`.
- Không có ICFG-PEDES thật hoặc checkpoint người dùng `best.pth` 1.3GB được mount.
- Không chạy Kaggle/GPU hay job 60 epoch. Không báo kết quả retrieval thực nghiệm.

## Kết quả thực tế

Lệnh suite cuối (basetemp là thư mục tạm mới trong workspace):

```powershell
.venv/Scripts/python.exe -m pytest tests -q --basetemp=.cache/pytest-final-complete --tb=short
```

**55 passed, 3 skipped in 4.90s.** Các lần đầu pytest bị sandbox Windows chặn
thư mục tạm; suite trên đã chạy lại thành công với quyền ghi thư mục test.

| Nhóm | Trạng thái | Quan sát |
|---|---|---|
| Frequency | PASS | 384×128, 224×224, kích thước lẻ và 1×1; dtype/device, finite, backward, gradient khác 0, train/eval, alpha=0, cache, Hermitian symmetry |
| FFT precision | PASS CPU | FP16/BF16 đầu vào, FFT FP32 trong autocast; output giữ dtype |
| Local | PASS | CNN pool/candidate shapes, softmax/top-k, gradients, nhiều kích thước, activation checkpoint tương đương |
| Fusion | PASS | Đủ 4 mode; đúng residual formula, alpha=0/disabled identity chính xác |
| IRRA integration | PASS | CLIP nhỏ nhưng code IRRA/Transformer/SDM/MLM/ID thật; train/inference dùng cùng ảnh, text nguyên, optimizer/LR, freeze |
| Checkpoint | PASS tổng hợp | Baseline payload có module-prefix; optimizer fresh khi warm-start; roundtrip/resume cả 4 mode; missing/unexpected/shape/partial/corrupt prompt/config báo lỗi |
| Metrics | PASS | Đối chiếu công thức upstream, chunk 1/4/30; absent IDs, relevance count, gallery nhỏ, percentage scaling |
| Training workflow | PASS | 1 batch × 1 epoch mỗi mode; best vẫn lưu khi R1=0, last/RNG/scheduler, JSON final đủ 5 metric |
| Dataset/split/config | PASS | Holdout identity-disjoint deterministic, YAML restore và CLI override, ambiguity checkpoint, layout ICFG |
| File-based smoke | PASS tổng hợp | JPEG + annotation nhân tạo → loader ICFG thật → resize/normalize → tokenizer 49408 → CLIP nhỏ → SDM/MLM/ID → backward/optimizer |
| Notebook | PASS cấu trúc/cú pháp | 12 cell; nbformat validate; mọi code cell compile được, không chứa kết quả chạy bịa |
| CUDA module tests | SKIPPED ×3 | local/frequency/both: CUDA device unavailable |
| ViT-B/16 thật + best.pth người dùng | NOT EXECUTED | Không có checkpoint/dataset; synthetic roundtrip không thay cho kiểm tra này |
| Kaggle end-to-end / VRAM T4 | NOT EXECUTED | Cần người dùng mount inputs và chạy notebook |

Các bước milestone đã thực thi: regression metric trước; frequency/checkpoint/
integration/split **25 passed** trước khi thêm local; sau local/fusion **39 passed**;
sau workflow **47 passed**; các kiểm tra config, smoke file, notebook và path
discovery và pool một candidate ở softmax đưa suite cuối tới 55 passed / 3 skip.

## Đối chiếu trực tiếp code upstream

```bash
python -m experiments.lgsp_tbpr.audit_baseline
```

Đã chạy thành công trên Git checkout này. Script đọc nguyên bản
`model/build.py`, `utils/metrics.py` từ base commit, thay duy nhất bộ tải CLIP
bằng CLIP random nhỏ, dùng cùng seed và đường FP16 cũ trên CPU:

- State keys và mọi state tensor giữa upstream và mode baseline: **bitwise bằng nhau**.
- Toàn bộ output SDM/MLM/ID/accuracy/temperature: **bitwise bằng nhau**.
- Embedding ảnh và text: **bitwise bằng nhau**.
- CMC/mAP/mINP full-matrix upstream và chunked: **tương đương số học**.

Script cần Git object của base commit, không chạy được từ ZIP không chứa `.git`.
Suite thông thường vẫn có regression identity/metric để dùng trong ZIP Kaggle.

## Số đo kỹ thuật quan sát được

Các số sau chỉ trên mô hình CLIP nhỏ random, **không phải ViT-B/16 pretrained**:

- Test near-identity: mean embedding cosine `1.000000000`,
  max absolute embedding difference `9.41753387e-06`.
- Saved-tensor hook của local, batch `[2,3,64,32]`: `2,349,828 bytes`;
  đây là tổng kích thước các tensor autograd yêu cầu lưu (có thể đếm lặp),
  không phải peak RSS hay VRAM.
- File-based smoke tổng hợp: image/text embedding `[2,64]`,
  SDM `18.44748306274414`, ID `1.0979573726654053`, MLM `10.446741104125977`,
  tổng loss `29.9921817779541`.
- Gradient norm frequency logits `1.553677248011809e-05`,
  alpha `0.01661200448870659`; finite và khác 0.
- Smoke near-identity: cosine `1.0`, max embedding difference
  `2.0772218704223633e-05`, max image difference `9.632110595703125e-05`.
- Frequency contribution/input norm ratio lúc init `0.0001250002242`.
- Một lượt CPU smoke ghi forward ~`0.0059s`, backward ~`0.0117s`;
  không warm-up/benchmark, không dùng để ước tính runtime Kaggle.

Chạy smoke trên Kaggle sẽ ghi lại chính các trường này với backbone/dữ liệu thật,
kèm peak VRAM, để đánh giá mức gần identity thực tế thay vì giả định.

## Kiểm tra bổ sung

- `git diff --check`: PASS.
- Compile các Python modules và cell notebook: PASS.
- Notebook schema `nbformat.validate`: PASS.
- `bash -n experiments/lgsp_tbpr/run_experiments.sh`: PASS (Git Bash).
- Source bundle được tạo từ working tree; manifest chứa SHA256 mỗi file.
  Đã kiểm tra SHA256/compile source của 65 file trong archive, giải nén rồi
  import `model`, `datasets`, `utils.options` và khởi tạo tokenizer đủ 49408 token
  từ thư mục đã giải nén. Đây không phải chạy Kaggle.

## Việc còn cần thực thi trên Kaggle

1. Mount ICFG-PEDES + baseline best.pth/configs.yaml + source ZIP.
2. Xác nhận split và baseline checkpoint thực, chạy baseline evaluation gốc.
3. Smoke frequency, kiểm tra finite loss/gradient, cosine và VRAM thực.
4. Fine-tune E2 rồi matched E0 cùng 5 epoch, cố định settings trước test cuối.
5. Nếu E2 ổn, chạy E1/E3; sau screening mới cân nhắc budget gốc và ≥3 seeds.

Chưa thể đánh dấu các bước GPU/dataset/checkpoint thật này đã hoàn tất.
