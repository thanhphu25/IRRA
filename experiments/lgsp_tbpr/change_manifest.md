# Danh sách thay đổi

11 file tracked sửa; 32 file mới, trên nhánh `feat/lgsp-tbpr`.

| File | Trạng thái | Mục đích |
|---|---|---|
| [datasets/build.py](../../datasets/build.py) | Sửa | Chọn val/test/holdout, giữ classifier label indices và pipeline ảnh cũ. |
| [model/build.py](../../model/build.py) | Sửa | Biến đổi ảnh chung train/inference/MLM, freeze, FP32 prompt masters, CPU dtype. |
| [processor/processor.py](../../processor/processor.py) | Sửa | Metric nhanh khi train, final đầy đủ; best/last, scalar meters, JSON và VRAM/time. |
| [solver/build.py](../../solver/build.py) | Sửa | Nhóm optimizer của prompt với learning rate riêng. |
| [solver/lr_scheduler.py](../../solver/lr_scheduler.py) | Sửa | Tránh chia 0 khi chạy budget ngắn bằng số epoch warmup. |
| [test.py](../../test.py) | Sửa | Restore strict checkpoint, override path/device và xuất metrics.json. |
| [train.py](../../train.py) | Sửa | Warm-start, resume epoch+1/RNG, seed, output riêng và lưu config. |
| [utils/checkpoint.py](../../utils/checkpoint.py) | Sửa | Strict key/shape/config validation, baseline warm-start, full resume và RNG. |
| [utils/logger.py](../../utils/logger.py) | Sửa | Append log khi resume, ghi UTF-8. |
| [utils/metrics.py](../../utils/metrics.py) | Sửa | Exact query-chunked ranking/AP/INP, API rank cũ và edge cases. |
| [utils/options.py](../../utils/options.py) | Sửa | Bốn mode, tham số prompt, YAML restore, chiến lược train và CLI validation. |
| [.gitignore](../../.gitignore) | Thêm | Bỏ qua môi trường, cache, checkpoint và bundle lớn. |
| [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) | Thêm | Attribution LGSP và giữ nguyên license MIT upstream. |
| [datasets/splits.py](../../datasets/splits.py) | Thêm | Train-only identity holdout deterministic. |
| [experiments/lgsp_tbpr/README.md](../../experiments/lgsp_tbpr/README.md) | Thêm | Research audit, adaptation, công thức, protocol và hướng dẫn chạy. |
| [experiments/lgsp_tbpr/audit_baseline.py](../../experiments/lgsp_tbpr/audit_baseline.py) | Thêm | Đối chiếu model/metric trực tiếp với Git base commit. |
| [experiments/lgsp_tbpr/bundle.py](../../experiments/lgsp_tbpr/bundle.py) | Thêm | Source ZIP từ working tree và SHA256 manifest, không cần push. |
| [experiments/lgsp_tbpr/change_manifest.md](../../experiments/lgsp_tbpr/change_manifest.md) | Thêm | Danh sách đầy đủ file thay đổi và mục đích. |
| [experiments/lgsp_tbpr/prepare_kaggle.py](../../experiments/lgsp_tbpr/prepare_kaggle.py) | Thêm | Tìm dataset/checkpoint rõ ràng, chuẩn bị symlink không ghi đè dữ liệu. |
| [experiments/lgsp_tbpr/requirements.txt](../../experiments/lgsp_tbpr/requirements.txt) | Thêm | Các dependency nhỏ; giữ torch/torchvision của Kaggle. |
| [experiments/lgsp_tbpr/run_experiments.sh](../../experiments/lgsp_tbpr/run_experiments.sh) | Thêm | Lệnh screening E0/E1/E2/E3 với matched settings. |
| [experiments/lgsp_tbpr/smoke_test.py](../../experiments/lgsp_tbpr/smoke_test.py) | Thêm | Một batch raw dataset, losses/gradients/features/cosine/time/VRAM. |
| [experiments/lgsp_tbpr/validation_report.md](../../experiments/lgsp_tbpr/validation_report.md) | Thêm | Kết quả test đã chạy và các bước chưa xác minh thực tế. |
| [model/lgsp/__init__.py](../../model/lgsp/__init__.py) | Thêm | Exports module LGSP. |
| [model/lgsp/frequency_prompt.py](../../model/lgsp/frequency_prompt.py) | Thêm | FFT FP32, smooth ring masks/cache, real và original-style residual. |
| [model/lgsp/local_prompt.py](../../model/lgsp/local_prompt.py) | Thêm | CNN pool, pixel cosine selection, top-k/dropout/recomputation. |
| [model/lgsp/spatial_fusion.py](../../model/lgsp/spatial_fusion.py) | Thêm | Residual fusion bốn mode và diagnostics tùy chọn. |
| [notebooks/irra_lgsp_icfg_kaggle.ipynb](../../notebooks/irra_lgsp_icfg_kaggle.ipynb) | Thêm | Workflow Kaggle hoàn chỉnh: inputs, smoke, E2/E0, full test, artifacts. |
| [pytest.ini](../../pytest.ini) | Thêm | Cấu hình test suite. |
| [tests/conftest.py](../../tests/conftest.py) | Thêm | Fixture CLIP nhỏ, batch tổng hợp; dùng IRRA và objectives thật. |
| [tests/test_checkpoint_loading.py](../../tests/test_checkpoint_loading.py) | Thêm | Warm-start/resume/roundtrip, config/key/shape corruption. |
| [tests/test_cuda.py](../../tests/test_cuda.py) | Thêm | CUDA FP16/autocast/gradients; skip khi thiếu GPU. |
| [tests/test_frequency_prompt.py](../../tests/test_frequency_prompt.py) | Thêm | Shapes/precision/symmetry/cache/gradients và original-style variant. |
| [tests/test_local_prompt.py](../../tests/test_local_prompt.py) | Thêm | Candidate pool, weights, top-k, backward và activation budget. |
| [tests/test_metrics.py](../../tests/test_metrics.py) | Thêm | Regression công thức gốc, chunks và retrieval edge cases. |
| [tests/test_model_integration.py](../../tests/test_model_integration.py) | Thêm | Train/inference/MLM, optimizer, freeze, text và near-identity. |
| [tests/test_notebook.py](../../tests/test_notebook.py) | Thêm | Cú pháp các cell Python, output/execution-count sạch. |
| [tests/test_options.py](../../tests/test_options.py) | Thêm | Config/resume restore, CLI override và kiểu argument. |
| [tests/test_prepare_kaggle.py](../../tests/test_prepare_kaggle.py) | Thêm | Phát hiện layout dataset và từ chối checkpoint mơ hồ. |
| [tests/test_smoke_pipeline.py](../../tests/test_smoke_pipeline.py) | Thêm | JPEG/annotation tổng hợp qua loader ICFG/tokenizer và smoke thật. |
| [tests/test_spatial_fusion.py](../../tests/test_spatial_fusion.py) | Thêm | Bốn công thức fusion và disable identity. |
| [tests/test_splits.py](../../tests/test_splits.py) | Thêm | Identity holdout disjoint và reproducible. |
| [tests/test_training_workflow.py](../../tests/test_training_workflow.py) | Thêm | Một epoch synthetic, best/last và metric JSON cả 4 mode. |

Hai file có sẵn do người dùng cung cấp `prompt.md` và
`irra-icfg-pedes-baseline.ipynb` vẫn giữ nguyên và chưa track.
Artifact tạo thêm (gitignore): `dist/irra_lgsp_source.zip`.
Môi trường `.venv`, reference/cache và checkpoint test nằm trong các đường dẫn đã ignore.
