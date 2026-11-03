# GROUP 9 — Hướng dẫn tự retrain model (`retrain_pipeline.py`)

Script chạy **offline trên máy bạn**, không chạy trong Streamlit. Nó làm đúng 7 bước trong `deployment_bundle/retraining_config.json` và **không bao giờ tự thay model đang chạy** nếu bạn không thêm `--promote` và mọi tiêu chí chưa đạt.

```
BRFSS 2015 (lịch sử) ─┐                         ┌─ locked test 50,736 dòng (tái tạo y hệt, không bao giờ train)
                      ├─ validate ─ kiểm policy ─┤
Sheet export / data mới┘   (loại dòng lỗi)       └─ train cũ + dòng mới (trừ holdout) ─► model ứng viên
                                                                                       │
                     so sánh ứng viên vs production trên CÙNG các dòng (PR-AUC, Brier, Precision@K)
                                                                                       │
                      đạt hết tiêu chí + có --promote ─► thay deployment_bundle (bản cũ được lưu archive)
```

## 1. Cài đặt (một lần)

Trong thư mục `diabetes_app`, Terminal VS Code:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-retrain.txt
.\.venv\Scripts\python.exe test_retrain_offline.py
```

Phải thấy 5 dòng `PASS`. Dùng Python 3.13 như app để kết quả tái lập chính xác.

## 2. Chuẩn bị dữ liệu

1. Tạo thư mục `data\` (đã nằm trong `.gitignore`, **không đẩy lên GitHub**).
2. Copy file gốc của nhóm vào: `data\diabetes_binary_health_indicators_BRFSS2015.csv` (253,680 dòng, không sửa gì).
3. Dữ liệu mới có nhãn — một trong hai dạng:
   - **Export Google Sheet**: mở Sheet `Predictions` → điền cột `actual_diabetes_outcome` (0 hoặc 1) và `outcome_recorded_at` (ISO, ví dụ `2026-11-20T10:00:00+00:00`) → **File › Download › CSV** → lưu `data\sheet_export.csv`. Dòng nào chưa có outcome sẽ được bỏ qua (pending), không bị coi là lỗi.
   - **Dataset mới đã recode** (ví dụ BRFSS 2024 của Hưng): CSV có đủ 21 cột feature + cột `Diabetes_binary`, cùng cách mã hoá như 2015. Phải kiểm tra tính tương thích trước khi dùng.

Lần sau chỉ cần export lại cả Sheet (file dài thêm). Script tự nhận biết dòng nào model hiện tại **đã dùng rồi** (qua `deployment_bundle/new_data_manifest.json`) và chỉ đếm dòng mới.

## 3. Chạy

**Chạy thử, không đụng vào model đang chạy** (nên làm trước):

```powershell
.\.venv\Scripts\python.exe retrain_pipeline.py --new-data data\sheet_export.csv
```

**Nếu kết quả ổn và muốn thay model** (tắt Streamlit trước, Ctrl+C):

```powershell
.\.venv\Scripts\python.exe retrain_pipeline.py --new-data data\sheet_export.csv --promote
```

Tuỳ chọn hay dùng:

| Tuỳ chọn | Tác dụng |
|---|---|
| `--new-data A.csv B.csv` | Nhiều file cùng lúc (Sheet export + dataset) |
| `--ignore-schedule` | Chạy sớm hơn lịch 30 ngày (vẫn được promote) |
| `--skip-cv` | Bỏ 5-fold CV để chạy nhanh hơn khoảng 2 phút (OOF PR-AUC sẽ để trống) |
| `--force` | Bỏ qua yêu cầu số lượng dữ liệu — **chỉ để test**, ứng viên không bao giờ được promote |
| `--historical PATH` | Đường dẫn khác cho file BRFSS 2015 |

## 4. Đọc kết quả

| Exit code / STATUS | Nghĩa |
|---|---|
| `2` · `SKIPPED_POLICY_NOT_MET` | Chưa đủ 30 ngày hoặc chưa đủ dòng có nhãn mới → không train |
| `0` · `CANDIDATE_READY` | Ứng viên đạt tiêu chí, nằm ở `model_registry\candidates\v…` — chưa deploy |
| `0` · `PROMOTED` | Đã thay `deployment_bundle\`; bản cũ ở `model_registry\archive\` |
| `3` · `REJECTED_CRITERIA_FAILED` | Ứng viên kém hơn → giữ model cũ |
| `1` · `ERROR` | Dữ liệu sai / không tái tạo được locked test → không làm gì cả |

Mỗi lần chạy tạo `retraining_runs\<run_id>\`: `report.md` (bảng so sánh để chép vào báo cáo), `report.json`, và `rejected_rows.csv` (dòng bị loại + lý do, số dòng khớp với số dòng trong Sheet).

Kiểm tra một ứng viên bằng smoke test: `.\.venv\Scripts\python.exe smoke_test.py model_registry\candidates\<thư_mục>`

## 5. Tiêu chí (trong `retraining_config.json`)

- **Đủ dữ liệu:** ≥ 500 dòng có nhãn chưa dùng, ≥ 50 dương tính, ≥ 50 âm tính; ≥ 30 ngày từ lần train trước.
- **Được promote khi:** PR-AUC ứng viên ≥ production trên holdout dữ liệu mới (hoặc locked test nếu holdout quá ít); PR-AUC locked test không giảm quá 0.005; Brier không tăng quá 0.002; Precision@K ở cả 6 mức capacity không giảm quá 0.02.

⚠️ Các con số này là **mặc định đề xuất**, đánh dấu `PROPOSED` trong file — cần **Hưng xác nhận** trước khi đưa vào báo cáo. Sửa trực tiếp trong JSON, không cần sửa code.

## 6. Rollback

Tắt Streamlit, xoá `deployment_bundle\`, copy thư mục `model_registry\archive\v<phiên_bản_cũ>_…` về và đổi tên thành `deployment_bundle`, chạy `smoke_test.py`.

## 7. Sau khi promote (làm tay)

- README §3 và caption trong `app.py` còn ghi **1.03%** → sửa theo `sample_output.json` mới (report.md ghi sẵn số mới).
- Đọc lại đoạn mô tả calibration trong `app.py` (“predicted scores are higher…”) so với đường cong mới.
- Commit `deployment_bundle\` mới lên GitHub để Streamlit Cloud dùng; báo người viết Methodology/Results/5.1.

## 8. Lưu ý khi viết báo cáo

- **Đã kiểm chứng:** trên dữ liệu gốc, pipeline tái tạo **chính xác** model v1.0.0 (chênh lệch điểm = 0), Brier 0.0987, OOF PR-AUC 0.4384 / ROC-AUC 0.8316, toàn bộ `risk_reference`, `capacity_config` và bảng calibration.
- Dữ liệu trong Sheet theo `LOGGING_SETUP_VI.md` là **dữ liệu giả lập cho demo** → không được retrain thật từ đó và không báo cáo đó là “model học từ người dùng”.
- `make_demo_prediction_log.py` tạo file `*_SIMULATED.csv` chỉ để thử pipeline; script **chặn promote** với file có chữ SIMULATED trong tên.
- Người dùng app không phải mẫu ngẫu nhiên của dân số (selection bias); outcome tự báo cáo. Đây là giới hạn cần nêu ở phần Discussion.
- Hyperparameter được **giữ nguyên** từ lần tối ưu Optuna của v1.0.0, không tuning lại.
