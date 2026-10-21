# SCFP V2: audit chẩn đoán, không retrain

Audit trên branch `feat/scfp-v2`, nguồn ban đầu `03ea2f8`. Chưa có checkpoint 60 epoch hoặc ICFG-PEDES trong workspace; chưa đo lại Rank-1 63.180%, gradient relevance 2.77e-9 hay ON/OFF của checkpoint đó. Các số này là thông tin người dùng cung cấp. Kết quả local dùng mô hình nhỏ ngẫu nhiên và fixture; không dùng để kết luận hiệu quả retrieval.

1. **Lỗi được xác nhận.** `text_content_mask` trong [semantic_slots.py](../../model/scfp/semantic_slots.py) loại mọi ID 0, dù tokenizer có token 0 là `!`. Ví dụ tokenizer thật mã hóa `!$` thành `[0, 259]`; token 0 nằm trước EOT vẫn bị bỏ. Audit đếm `content_id_zero_count` và `excluded_non_special_content_count`; chưa biết mức xuất hiện trong ICFG-PEDES. Không sửa mask/model trong đợt này. Chuỗi lỗi `Unsafe intervention ... identity-compatible` không có trong source hiện tại; nó không chứng minh lỗi forward của model. Diagnostic mới dùng đúng `forward(tokens, grid_size)` và cờ `enabled` đã có, chỉ trong eval/no_grad, khôi phục bằng `finally`.

Đã sửa lỗi đóng gói source trên Windows: manifest ZIP trước đây dùng dấu `\`, không khớp đường dẫn POSIX khi kiểm hash trên Kaggle/Linux. `bundle.py` nay dùng `as_posix()`; không đổi kiến trúc, loss hay trọng số.

2. **Frequency: cấu trúc và giới hạn xác nhận.** [feature_frequency.py](../../model/scfp/feature_frequency.py) dùng bán kính `sqrt((fy/.5)^2+(fx/.5)^2)/sqrt(2)` trên FFT feature grid, không phải FFT ảnh gốc. Bốn mask binary rời nhau, tổng bằng 1 ở mỗi bin; không chuẩn hóa theo diện tích băng. FFT/IFFT dùng `norm='ortho'`, giữ đối xứng liên hợp, CLS được copy nguyên tại vị trí chèn sau block chỉ định. Các block sau vẫn có thể làm CLS thay đổi.

| Index | Khoảng bán kính chuẩn hóa | Vai trò tương đối | Số bins trên grid 24×8 | Gain suy ra từ số người dùng cung cấp |
|---|---|---|---:|---:|
| 0 | [0, .25) | thấp, gồm DC | 19 | 0.97860 |
| 1 | [.25, .5) | trung thấp | 50 | 1.03124 |
| 2 | [.5, .75) | trung cao | 92 | 0.97685 |
| 3 | [.75, 1] | cao, gồm góc/Nyquist | 31 | 0.97729 |

`weights=softmax(logits/temperature)`, `alpha=tanh(alpha_raw)`, `gain_b=1+alpha*tanh(4*weights_b−1)`. `alpha_raw=0.03155` cho alpha≈0.0315395. Bảng là phép thế các số làm tròn được cung cấp, không phải phép đo checkpoint; gain trung bình thực tế phải tính theo từng ảnh vì tanh phi tuyến. Softmax thấp không có nghĩa băng bị xóa; chưa thấy lỗi coverage hoặc suppression ngoài công thức. Băng nhiều bins có thể chứa nhiều năng lượng hơn; mask không cân bằng năng lượng và không được thiết kế để làm vậy.

Gate chỉ nhận `grid.mean(H,W)` qua LayerNorm/MLP. Perturbation tần số có trung bình bằng 0 có thể thay đổi phổ mà không đổi gate: kiểm thử đã tái hiện. Đây là giới hạn descriptor, chưa chứng minh gate không thích nghi trên dữ liệu thật. Audit ghi per-image logits/weights/gains, std/range giữa ảnh, năng lượng đầu vào/đầu ra, residual và response 2D thật. `frequency_response.png` hiển thị mask, gain, RMS từng băng và residual, không chỉ softmax.

3. **Slots và gradient: giới hạn xác nhận, giả thuyết còn mở.** [semantic_slots.py](../../model/scfp/semantic_slots.py) dùng query chung nhưng attention/projection riêng hai modality; không có nhãn thuộc tính. [local_alignment.py](../../model/scfp/local_alignment.py) tính cosine cùng index và relevance của caption, dùng mọi cặp cùng PID làm positive ở cả hai chiều. Kiểm thử với PID lặp xác nhận target đúng. Mask loại SOT/EOT/MASK và vị trí sau EOT; empty text dùng sentinel, attention nội dung bằng 0. Chú ý lỗi ID 0 ở mục 1.

[diversity.py](../../model/scfp/diversity.py) phạt bình phương cosine off-diagonal của embeddings; không trực tiếp phạt attention trùng nhau. Tại collapse hoàn toàn, cosine=1 nhưng gradient theo embeddings bằng 0: đã xác minh bằng autograd. Cosine 0.8812/0.8307 được cung cấp cho thấy slot chưa tách mạnh, không tự chứng minh attention trùng hoặc penalty hoàn toàn vô ích. Một snapshot không đo được hiệu quả regularization trong cả quá trình train.

Relevance là `raw=.1+.9*sigmoid(head(text_slot))`, sau đó chia tổng. Floor ngăn nghiệm toàn 0, không bảo đảm relevance có thông tin. Nếu các điểm slot giống nhau, thay relevance không đổi local score; kiểm thử xác nhận. Gradient 2.77e-9 cũng có thể do SDM bão hòa trên batch dễ, sigmoid bão hòa hoặc nhiều yếu tố kết hợp. Margin 0.4329 đơn lẻ không phân biệt được chúng. Audit ghi phân bố logits/raw/weights, sigmoid derivative, chênh lệch điểm khi thay relevance bằng uniform, độ lệch điểm giữa slot, negative probability mass và đạo hàm qua từng mắt xích. Chưa gán nguyên nhân khi chưa có kết quả thật.

Trong [build.py](../../model/build.py), slot loss nhận patch/text features có graph; các bản detach chỉ phục vụ log. Audit dùng **autograd.grad trên backbone dùng chung**, đo norm và cosine `slot_loss`/`sdm_loss` theo modality và từng parameter, kèm weighted diversity norm/cosine. Nó kiểm tra loss tính lại khớp production trong tests. Final CLS/EOT có direct slot gradient bằng 0 do slice/mask, nhưng các tham số backbone ở trước vẫn nhận gradient; kiểm thử xác nhận cả visual và text. Nonzero trên fixture chỉ chứng minh đường graph tồn tại, không chứng minh auxiliary loss hữu ích trên checkpoint đã train. Gradient mặc định đo eval, resize/normalize cố định, batch 8 để giảm VRAM; không đại diện mọi training batch 64. Precision gốc được ghi; có thể chạy một audit riêng `--precision fp32` để kiểm tra underflow, không lưu trọng số cast.

4. **Đề xuất V2.1 sau khi đọc kết quả thật.** Đếm/tái hiện ảnh hưởng lỗi mask trước. Nếu gain ít biến thiên giữa ảnh, xem xét descriptor có năng lượng băng tần. Nếu attention chồng lấn, thử riêng ràng buộc attention hoặc cạnh tranh slot; nếu relevance gần uniform và ít ảnh hưởng, thử đối chứng trọng số đều. Nếu local loss bão hòa hoặc gradient xung đột global SDM, kiểm tra batch/scale trên train hoặc validation phù hợp trước khi chọn thay đổi. Mỗi thử nghiệm cần đối chứng, nhiều seed và budget bằng nhau; ON/OFF inference không thay thế frequency-only ablation được train riêng. Chưa triển khai V2.1/V3 hay chọn hyperparameter bằng official test.

5. **Cách chạy Kaggle.** Dùng [notebook audit](../../notebooks/irra_scfp_v2_audit_kaggle.ipynb), attach checkpoint V2 và ICFG-PEDES. Để dùng source chưa push, tạo ZIP từ workspace rồi attach cùng notebook:

```bash
python -m experiments.scfp.bundle --output IRRA_scfp_v2_audit_source.zip
```

Hoặc chạy các lệnh sau trong terminal/cell `%%bash` Kaggle, sau khi đặt ba đường dẫn đúng mount của bạn. `CHECKPOINT` phải trỏ tới V2, không phải baseline. Thư mục source/output bên dưới phải chưa tồn tại.

```bash
SOURCE_ZIP=/kaggle/input/YOUR_SOURCE/IRRA_scfp_v2_audit_source.zip
CHECKPOINT=/kaggle/input/YOUR_V2_RUN/best.pth
python -m zipfile -e "$SOURCE_ZIP" /kaggle/working/IRRA_scfp_v2_audit
cd /kaggle/working/IRRA_scfp_v2_audit
python -m pip install -r experiments/scfp/requirements-audit.txt
python -m experiments.scfp.prepare_kaggle --input_root /kaggle/input --data_root /kaggle/working/scfp_audit_data
python -m experiments.scfp.audit \
  --checkpoint "$CHECKPOINT" \
  --root-dir /kaggle/working/scfp_audit_data \
  --output-dir /kaggle/working/SCFP_V2_AUDIT/results \
  --device cuda --precision checkpoint --seed 2026 \
  --num-identities 32 --images-per-identity 4 --examples 6 \
  --batch-size 16 --gradient-batch-size 8 --gradient-batches 3
```

Nếu checkpoint không nhúng config, thêm `--config-file /kaggle/input/YOUR_V2_RUN/configs.yaml` đúng run. Nếu có nhiều bộ dữ liệu, thêm `--annotation_source` và `--images_source` cho prepare_kaggle. Không tự chọn ngẫu nhiên khi discovery không duy nhất. Script không cần mạng sau khi đủ source/dependencies/data/checkpoint. CUDA/T4 và checkpoint thật chưa được chạy tại workspace này.

Outputs gồm bốn JSON được yêu cầu, ba PNG được yêu cầu, `SCFP_V2_AUDIT.md` và `subset_manifest.json`. Manifest lưu index annotation/caption/PID, danh sách mẫu minh họa và batch gradient; cùng hash source/checkpoint/config/precision trong JSON để tái lập. Sáu hình mẫu được chọn **trước inference** theo độ dài caption và độ phủ PID; không chọn theo kết quả đẹp. Attention được lấy trung bình qua các head, không gán tên bộ phận cơ thể. Thang màu chung giữa bốn slot trong từng cặp.

R1 ON/OFF là text-to-image trên **fixed training subset đã được mô hình thấy**, có paired image trong gallery; báo cùng cosine global image embedding, mean/max difference và lặp ON để kiểm tra tái lập. Không so trực tiếp R1 tập con với official Rank-1 63.180%/63.573%. Phép đo không thực hiện được có `unavailable`/`partial`, nguyên nhân và exit code 2; PNG không đo được ghi UNAVAILABLE. Lỗi đầu vào/load checkpoint ở preflight dừng sớm. Không optimizer, không ghi checkpoint; hash parameters/buffers và bytes checkpoint trước/sau được kiểm tra. Kết quả giả lập local nằm trong `reports/audit_synthetic/`, có nhãn SYNTHETIC rõ ràng.

Kiểm chứng local cuối: **38 passed**, PyTorch 2.11.0+cpu, Python 3.10; không CUDA. Bao gồm đối chiếu loss production, shared-parameter gradient, bypass FP32/FP16/BF16 trên grid chẵn/lẻ, Parseval/coverage/đáp ứng gain, tokenizer ID 0, checkpoint/data fixture, trường hợp unavailable, notebook và ZIP source. Lệnh chọn nhóm tests (không chạy training/optimizer.step):

```bash
python -m pytest -q tests/test_scfp_audit.py tests/test_feature_frequency.py tests/test_semantic_slots.py tests/test_local_alignment.py tests/test_model_integration.py tests/test_notebook.py
```

Môi trường local dùng dependency phụ trong `.venv/auditdeps`, tắt pytest plugin tự nạp và đặt temp dưới workspace để tránh ACL sandbox Windows. Đây không phải xác nhận chạy T4 hoặc checkpoint thật.
