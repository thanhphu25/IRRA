# IRRA + LGSP-inspired Local–Global Spatial Prompting

Mục tiêu: kiểm tra liệu spatial prompting phía ảnh có cải thiện TBPR, giữ nguyên
CLIP text encoder, embedding retrieval và `L_SDM + L_MLM + L_ID` của IRRA.
Chưa có số đo ICFG-PEDES thật; không suy diễn cải thiện Rank-1 từ test kỹ thuật.

## Nguồn và audit

- IRRA: [Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image
  Person Retrieval](https://arxiv.org/abs/2303.12501), CVPR 2023;
  [upstream](https://github.com/anosorae/IRRA), [fork](https://github.com/thanhphu25/IRRA).
  Tên “A Task-Oriented Framework…” trong prompt không phải tiêu đề của arXiv được dẫn.
- LGSP: [Revisiting Pool-based Prompt Learning for Few-shot Class-incremental
  Learning](https://arxiv.org/abs/2507.09183), ICCV 2025;
  [code](https://github.com/Jywsuperman/LGSP/tree/fe2d0619a6a023450a25cf6941de451af8c869d9).
- Base IRRA: `cfc23785545ead82dee2749fa0fc0ad948c96430`. Fork tại thời điểm audit
  trùng HEAD upstream (đã kiểm tra remote). Nhánh làm việc: `feat/lgsp-tbpr`.
  Ban đầu chỉ có hai file người dùng chưa track: `prompt.md`, notebook baseline;
  cả hai được giữ nguyên. Implementation nằm trên nhánh `feat/lgsp-tbpr`;
  commit/push chỉ thực hiện theo yêu cầu của người dùng.
- Đã đọc toàn bộ 14 cell notebook baseline. Notebook không chứa kết quả đo
  baseline, chỉ có bảng số liệu paper; vì vậy chưa biết baseline thực của người dùng.
- LGSP audit: `models/base/ViT_Network.py` chứa CNN pool, cosine theo pixel,
  `get_prompts`, `get_Frequency_mask` và lời gọi trong `forward`;
  `models/prompt.py` có nhánh CLS/key FFN và Gumbel softmax;
  `models/model_My.py` chủ yếu đăng ký ViT, không phải phần prompting chính.
  Giữ license MIT trong [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

Các điểm tích hợp đã kiểm tra:

| Thành phần | Hợp đồng thực tế / quyết định |
|---|---|
| `model/build.py` | Khởi tạo CLIP bằng `build_CLIP_from_openai_pretrained`; prompt gắn trong wrapper IRRA |
| `datasets/build.py` | Resize mặc định 384×128, augmentation cũ; mean `[0.48145466,0.4578275,0.40821073]`, std `[0.26862954,0.26130258,0.27577711]` |
| Đầu vào visual | `[B,3,H,W]` đã CLIP-normalize; CLIP tự chuyển sang dtype conv1 |
| Train | `transform_image` → `base_model(images, caption_ids)`; lấy CLS cho retrieval/ID |
| Inference | Cùng `transform_image` → `base_model.encode_image`; lấy CLS, float32 |
| MLM | Dùng toàn bộ visual tokens của đúng ảnh đã biến đổi; text/MLM objectives giữ nguyên |
| Optimizer | Nhóm từng parameter; prompt có `lgsp_lr` riêng, sau quy tắc bias/classifier cũ |
| Checkpoint | Payload `model`, `optimizer`, `scheduler`, `epoch`; mới thêm config, RNG, best score |
| Cấu hình | `configs.yaml`; test đọc config; train `--config_file` hoặc resume tự đọc config bên cạnh checkpoint |
| Best | Rank-1 validation; test cuối dùng đầy đủ metric; vẫn lưu best nếu Rank-1 bằng 0 ở lần đầu |
| Split | Mặc định CLI vẫn `test` như lịch sử; có `val` và train-only `holdout`; notebook chọn auto |
| CLIP source | Không sửa `model/clip_model.py`, `model/objectives.py`, `model/__init__.py` |

## Phép chuyển LGSP sang TBPR

Đặt `X` là ảnh đã chuẩn hóa. Mọi branch **trả scaled residual**, không trả ảnh:

```text
baseline:  Y = X                     (trả đúng cùng tensor, không module phụ)
local:     Y = X + alpha_l P_l(X)
frequency: Y = X + alpha_f P_f(X)
both:      Y = X + alpha_l P_l(X) + alpha_f P_f(X)
```

Cả hai residual trong `both` được tính từ **cùng X**, không chạy tuần tự.
Alpha khởi tạo `0.001`, khác 0 để gradient đến trọng số bên trong ngay bước đầu.
Ảnh không denormalize/re-normalize, không clamp. FFT/CNN prompt tính FP32;
branch và fusion trả cùng dtype/device/shape đầu vào. Backbone giữ quy tắc
FP16 của IRRA trên CUDA, hoặc FP32 với `--precision fp32` / `--device cpu`.

| Cơ chế LGSP gốc | Bản TBPR | Lý do |
|---|---|---|
| Pool nhiều CNN hai lớp, kernel có thể lớn | Mặc định 4 CNN `3→8→3`, kernel 3, ReLU, dropout tùy chọn | GPU đơn, ảnh người 384×128 |
| `get_prompts` có cosine channel-wise theo pixel và softmax theo pool | Giữ cơ chế cosine theo pixel, temperature chia; tùy chọn top-k theo pixel | Không cần thêm một lượt CLIP hay prototype lớp |
| Nhánh khác dùng CLS, key FFN, Gumbel, prototype từ classifier, FSCIL session | Không đưa vào; không tái tạo/đăng ký Parameter mới trong forward | Retrieval không có incremental classes; optimizer ổn định |
| Dropout trong generator | Expose dropout, mặc định 0 để kiểm soát ablation | So sánh train/inference deterministic khi không dropout |
| Ring sigmoid; radii init dựa trên 224×224; ring đầu gần rỗng | `real`: K ring trơn partition-of-unity, kể cả DC/corner, radii theo H,W thực | Tránh ring chết, hợp ảnh chữ nhật/kích thước lẻ |
| `softmax(weights * temperature)` | `real`: `softmax(logits / T)`, logits=0; `lgsp_original`: nhân T | Không âm thầm đổi nghĩa temperature giữa hai variant |
| `abs(IFFT(F + F*M))`, rồi `X + 0.1*(abs(...) - X)` | Mặc định `real(IFFT(F*M))` làm residual, alpha nhỏ học được | Giữ dấu pixel CLIP-normalized, gần identity |
| Spatial fusion code có cả sequential và alpha/beta mix hai ảnh | Residual song song, chỉ cộng X một lần | Ablation độc lập, không scale ảnh gốc ngoài ý muốn |
| ViT/VPT-Deep + classifier/FSCIL | IRRA CLIP dual encoder, SDM/MLM/ID gốc | Cô lập giả thuyết chuyển cơ chế spatial prompting |

Frequency mặc định:

```text
d(y,x) = sqrt((y-floor(H/2))² + (x-floor(W/2))²)
disk_i = sigmoid(4 * (radius_i - d))
rings = [disk_1, disk_2-disk_1, ..., 1-disk_(K-1)]
w = softmax(logits / T); M = sum_i w_i * ring_i
P_f = real(IFFT2(ifftshift(fftshift(FFT2(X)) * M)))
```

Khoảng cách dùng đơn vị bin phổ như code LGSP, không chuẩn hóa riêng hai trục;
đây không phải filter đẳng hướng theo cycles/pixel khi H≠W. Mask thực đối xứng
theo các cặp Fourier liên hợp, kể cả kích thước lẻ/Nyquist, nên phần ảo chỉ có
sai số số học. Cache một bộ ring theo H,W/device, không lưu cache vào checkpoint.
Logits đồng đều ban đầu cho `P_f=X/K`; gradient logits vẫn học được khi dữ liệu
có nhiều thành phần tần số. Chỉ một ring không đủ để kiểm nghiệm weighting nên K≥2.

`--frequency_variant lgsp_original` giữ phép biến đổi abs của code gốc:
`P_f = abs(IFFT(F*(1+M))) - X`, dùng ring sigmoid kiểu gốc, radii theo kích thước
thực và alpha học được. Đặt alpha 0.1 để khảo sát hệ số gốc, nhưng đây **không**
phải tái lập toàn bộ LGSP: backbone, domain chuẩn hóa, khởi tạo logits, radii,
loss và pipeline đều có khác biệt đã nêu. Với pixel âm, abs có thể đổi dấu và
tạo thay đổi lớn; chỉ dùng như ablation riêng.

Local: `P_i=CNN_i(X)`, `s_i(y,x)=cos_channel(X,P_i)/T`,
`w=softmax_pool(s)`, `P_l=sum_i w_i*P_i`. Top-k tùy chọn mask logits trước softmax;
chỉ candidate được chọn tại pixel nhận gradient ở pixel đó. Top-k=1 không có
gradient qua softmax selector, nhưng generator vẫn nhận gradient trực tiếp.
Không khẳng định prompts tương ứng đầu/thân/chân. Không có text-region loss.
Không stack thêm bản sao toàn bộ candidate pool trong phép tổng có trọng số;
`--local_checkpoint` giảm activation lưu bằng recomputation (PyTorch ≥2.1).

## Cấu hình

| Option | Mặc định | Ý nghĩa |
|---|---:|---|
| `--lgsp_mode` | baseline | baseline/local/frequency/both |
| `--frequency_num_rings` | 8 | Số ring, ≥2 |
| `--frequency_temperature` | 1 | Temperature >0 |
| `--frequency_alpha_init` | 0.001 | Residual frequency ban đầu |
| `--frequency_variant` | real | real / lgsp_original |
| `--local_pool_size` | 4 | Số CNN độc lập |
| `--local_hidden_dim` | 8 | Hidden channels |
| `--local_kernel_size` | 3 | Kernel lẻ, giữ H,W |
| `--local_selection` | softmax | softmax / topk, theo pixel |
| `--local_top_k` | 2 | Trong [1,pool_size], dùng khi topk |
| `--local_temperature` | 1 | Cosine temperature >0 |
| `--local_dropout` | 0 | Dropout trong generator |
| `--local_alpha_init` | 0.001 | Residual local ban đầu |
| `--local_checkpoint` | false | Recompute activation CNN khi backward |
| `--lgsp_lr` | 0.001 | LR prompt, riêng với IRRA `--lr` |
| `--freeze_backbone` | false | Freeze toàn bộ IRRA cũ, chỉ prompt trainable; baseline không hợp lệ |
| `--lgsp_diagnostics` | false | Ring weights, alpha, norm/ratio, utilization theo batch |
| `--init_checkpoint` | rỗng | Warm-start model, optimizer/scheduler mới |
| `--resume --resume_ckpt_file` | false/rỗng | Restore model/optimizer/scheduler/RNG, tiếp từ epoch+1 |
| `--config_file` | rỗng | Đọc YAML, CLI override; resume tự đọc YAML cạnh checkpoint |
| `--eval_chunk_size` | 128 | Query chunk cho evaluation |
| `--seed --deterministic` | 1/false | Seed Python/NumPy/Torch; tắt cuDNN benchmark khi bật |
| `--precision --device` | fp16/cuda | FP32 tùy chọn; CPU luôn FP32 backbone |
| `--val_dataset` | test | Giữ lịch sử; nên chọn val/holdout cho nghiên cứu mới |
| `--holdout_fraction --holdout_seed` | 0.1/42 | Identity holdout từ train, lưu danh sách ID trong config |

Số parameter mới mặc định: frequency 9; local 1,773; both 1,782.
Mỗi run có path timestamp tới microsecond, không ghi đè run khác.
RNG trước train được reseed sau khởi tạo để pool CNN không đổi thứ tự batch E0/E2.
Không bảo đảm bitwise giữa phần cứng, phiên bản CUDA/PyTorch khác nhau.

## Checkpoint và baseline

`Checkpointer.warm_start` nhận payload IRRA hoặc tensor state_dict; bỏ tiền tố
`module.` đúng từng key; chỉ cho phép thiếu **toàn bộ module prompt mới** từ
checkpoint baseline. Log missing/unexpected; thiếu IRRA keys, partial prompt,
unexpected keys và shape sai đều báo lỗi. Không dùng suffix matching rộng như
loader cũ vì có thể che giấu checkpoint hỏng. `load`/`resume` luôn yêu cầu đủ
prompt. Metadata LGSP và các option ngoài tensor được kiểm tra khi restore.
`torch.load(weights_only=True)` không fallback sang pickle tùy ý.

Warm-start không load optimizer, scheduler, epoch hay RNG. Resume tự lấy config
bên cạnh checkpoint, tiếp epoch đã lưu +1, giữ best score; không đổi tổng budget
scheduler khi resume. Muốn thêm 5 epoch từ model tốt nhất hãy mở run warm-start
mới. Nếu run ở Kaggle Input, copy cả thư mục vào Working trước khi resume.

```bash
python train.py --resume --resume_ckpt_file /kaggle/working/<run>/last.pth
```

Tái tạo baseline lịch sử (từ CLIP, không truyền init checkpoint):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --name irra_icfg_paper --lgsp_mode baseline \
  --img_aug --batch_size 64 --MLM --loss_names sdm+mlm+id \
  --dataset_name ICFG-PEDES --root_dir /kaggle/working/data --num_epoch 60 \
  --val_dataset test --output_dir /kaggle/working/IRRA_runs
python test.py --config_file /kaggle/working/<run>/configs.yaml
```

Không chạy 60 epoch trong môi trường phát triển. Baseline prompt bypass là
identity chính xác, loss functions/text encoder nguyên bản. Sửa hỗ trợ CPU bằng
dtype classifier thực, logit_scale chuyển cùng device qua nonpersistent buffer.
Lưu thêm last checkpoint/RNG, metric JSON và scalar meter đã detach để không giữ
autograd graph. Scheduler tránh chia 0 khi screening chỉ có 1–5 epoch.

## Split và evaluation

ICFG loader đọc `ICFG-PEDES.json` (khác ví dụ README upstream dùng underscore).
Notebook cũ hỗ trợ cả hai tên và symlink thành đúng tên; workflow mới giữ cách này.
`train`, `test`, các split còn lại được loader xem là `val`. Dữ liệu Kaggle thật
chưa được mount để xác nhận số mẫu/split; notebook in thống kê trước khi chạy.
Nếu có val riêng, dùng val. Nếu không: `--val_dataset holdout` chọn 10% identity
từ train bằng NumPy RandomState(42), loại toàn bộ ảnh/caption của chúng khỏi
continued training; không đổi index lớp/size classifier để load checkpoint cũ.
Danh sách ID lưu trong `configs.yaml`. Protocol phải giống nhau cho cả E0–E3.

**Giới hạn của holdout warm-start:** best.pth cũ đã nhìn thấy mọi train identity,
kể cả holdout này. Đây là screening control cho bước fine-tune, không phải val
chưa từng thấy xuyên suốt quá trình pretrain. Thí nghiệm cuối muốn holdout độc
lập cần train lại E0/E2 từ CLIP, loại cùng các identity trước mọi bước huấn luyện.
Không coi holdout này chữa được test-selection của checkpoint gốc. Giữ protocol
lịch sử `--val_dataset test` để đối chiếu, nhưng không dùng để tuning mới.

Training chỉ exact R1/R5/R10 (`topk` theo từng query chunk); final test full
argsort của **toàn bộ gallery cho từng query chunk**, tính đúng AP/INP, không
xấp xỉ bằng top-k. Không tạo similarity Q×G đầy đủ trong Evaluator.
Metric theo phần trăm. Query ID không có trong gallery tính 0 và vẫn trong mẫu
số (như notebook); rank quá gallery size được pad CMC cuối. Gallery/query rỗng
báo lỗi. Với các điểm similarity bằng nhau, thứ tự phụ thuộc PyTorch topk/argsort
như code gốc; regression metric dùng score phân biệt để tránh tie không xác định.

## E0/E1/E2/E3 và Kaggle

Đầu tiên chạy E2, sau đó E0 với cùng 5–10 epoch bổ sung. E1/E3 sau khi xác nhận
E2 ổn định. Thay batch size phải thay control tương ứng: SDM dùng quan hệ trong
batch nên accumulation không tương đương batch 64 thực.

| ID | Mode | Khởi tạo screening | Vai trò |
|---|---|---|---|
| E0 | baseline | Cùng best.pth | Control thêm cùng budget fine-tune |
| E1 | local | Cùng best.pth + CNN mới | Local contribution |
| E2 | frequency | Cùng best.pth + FFT weights mới | Chạy trước |
| E3 | both | Cùng best.pth + cả hai | Tương tác local/global |

1. Bật GPU đơn T4, Internet. Tạo bundle từ working tree: `python -m experiments.lgsp_tbpr.bundle`.
   Upload `dist/irra_lgsp_source.zip` làm Kaggle Input cùng dataset và output chứa
   baseline `best.pth`, ưu tiên kèm `configs.yaml`. ZIP cho phép dùng source
   trực tiếp; notebook cũng có thể clone research branch khi bật tùy chọn.
2. Import [notebook](../../notebooks/irra_lgsp_icfg_kaggle.ipynb), chỉnh cell config.
   Auto-detection từ chối chọn bừa khi nhiều annotation/checkpoint. Khi đó đặt
   `ANNOTATION_SOURCE`, `IMAGES_SOURCE`, `INIT_CHECKPOINT`, `BASE_CONFIG` cụ thể.
   Chỉ tạo symlink trong Working; không xóa thư mục/đổi mount người dùng.
3. Notebook kiểm tra source SHA256, cài dependency nhỏ, in split, chạy unit tests,
   đánh giá checkpoint baseline gốc, smoke test một batch, E2 rồi E0, test cuối.
   Baseline đã có config riêng sẽ lấy architecture/loss weights cũ; CLI settings
   thí nghiệm override rõ. Nếu không có config, giả định notebook cũ: ViT-B/16,
   CMT depth 4, MLM vocabulary 49408; shape mismatch sẽ dừng thay vì bỏ qua.
4. Đổi `EXPERIMENT='local'/'both'` cho E1/E3; giữ seed/budget/split/optimizer.
   Có `FREEZE_BACKBONE=True` cho chẩn đoán prompt-only; tắt matched full-FT
   control ở protocol này. Muốn từ CLIP thay vì IRRA, bỏ `--init_checkpoint` ở CLI.
5. Save Version, lấy artifacts ở `OUTPUT_ROOT`, source path và run paths được in
   chính xác. `metrics.json` và CSV chứa kết quả thực sau khi chạy, chưa điền số.

CLI trong Kaggle sau khi source/data đã chuẩn bị:

```bash
export INIT_CHECKPOINT=/kaggle/input/<attached-output>/best.pth
export DATA_ROOT=/kaggle/working/data
export EPOCHS=5 BATCH_SIZE=64 SEED=42 VAL_DATASET=holdout
# Nếu baseline có config riêng, đặt BASE_CONFIG để giữ architecture/loss weights:
# export BASE_CONFIG=/kaggle/input/<attached-output>/configs.yaml
bash experiments/lgsp_tbpr/run_experiments.sh frequency  # E2 đầu tiên
bash experiments/lgsp_tbpr/run_experiments.sh baseline   # E0 control
bash experiments/lgsp_tbpr/run_experiments.sh local      # E1
bash experiments/lgsp_tbpr/run_experiments.sh both       # E3
python test.py --config_file /kaggle/working/<new-run>/configs.yaml
```

Smoke test thực, không cached features:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.lgsp_tbpr.smoke_test \
  --img_aug --MLM --loss_names sdm+mlm+id --dataset_name ICFG-PEDES \
  --root_dir /kaggle/working/data --batch_size 64 --num_workers 2 \
  --val_dataset holdout --lgsp_mode frequency --init_checkpoint "$INIT_CHECKPOINT" \
  --seed 42 --output_dir /kaggle/working/IRRA_LGSP_runs/smoke_frequency
```

Mỗi run: `configs.yaml`, `train_log.txt`, `best.pth`, `last.pth`, TensorBoard,
`training_stats.jsonl` (thời gian/VRAM/loss/diagnostics nếu bật),
`validation_metrics.jsonl`; test tạo `test_log.txt`, `metrics.json`.
Notebook thêm `protocol.json`, `environment.txt`, `summary_*.csv`, smoke JSON.
Checkpoint có thể ~1.3GB như bản cũ; best+last cần nhiều disk. Chưa đo VRAM T4.
Nếu OOM, giảm evaluation batch/chunk trước; local có thể bật recomputation.
Không cài environment LGSP cũ. Kaggle dùng torch/torchvision sẵn có; Python≥3.9,
PyTorch≥2.1 khuyến nghị. Internet cần cho dependency và tải CLIP lần đầu;
offline phải gắn wheel dependencies và CLIP cache `~/.cache/clip/ViT-B-16.pt`
đúng checksum OpenAI, hoặc dùng `--pretrain_choice` trỏ file CLIP pretrained.

## Tests, giới hạn và kế hoạch

```bash
python -m pytest tests -q
```

Tests dùng IRRA thật, CLIP transformer thu nhỏ random (64-d, 1 layer), SDM/ID/MLM
thật; không tải ViT-B/16 trong CPU suite. Có kiểm tra train/inference dùng cùng
ảnh, text không đổi, optimizer coverage/LR, freeze, checkpoint roundtrip tất cả
mode, key/shape/config lỗi, metrics đối chiếu công thức cũ, split holdout,
one-batch training/evaluation. CUDA tests tự skip khi thiếu GPU.
Kết quả chạy thực và giới hạn xác minh ghi tại [validation_report.md](validation_report.md).

Diagnostics tùy chọn: ring weights, alpha, residual/contribution norm và ratio;
local mean selection weights, histogram prompt thắng theo pixel. Các số là
batch gần nhất, không diễn giải thành bộ phận cơ thể. Có API `frequency_mask`
để export mask/ảnh khác biệt nếu cần; không ghi ảnh mỗi batch.

Chưa kiểm chứng: ViT-B/16 thật + baseline best.pth 1.3GB, ICFG dữ liệu thật,
Kaggle CUDA/T4, runtime/VRAM và hiệu quả retrieval. Không tự launch full training.
Nếu screening có tín hiệu: từ cùng CLIP init chạy budget gốc 60 epoch, protocol
split thống nhất và ≥3 seeds; báo mean/std đủ R1/R5/R10/mAP/mINP và chi phí.
Không dùng cached global embeddings thay raw image train. Không có FSCIL,
continual learning, text-guided masks, text prompts, loss mới hoặc segmentation.
