# GROUP 9 — Logging Google Sheets MIỄN PHÍ (không thẻ ngân hàng)

Hướng dẫn này thay hoàn toàn phương án Google Cloud Service Account trước đó.
**Không cần Google Cloud Console, billing, credit card, service account hay file JSON key.**
Chỉ dùng dữ liệu **giả lập** cho demo; không nhập thông tin bệnh nhân thật.

## 1. Tạo Sheet (Google Sheets cá nhân)

1. Mở https://sheets.new (đăng nhập Gmail cá nhân).
2. Đổi tên file: `Group 9 - Diabetes Prediction Log`.
3. Đổi tên tab dưới cùng từ `Sheet1` thành **`Predictions`** (đúng chính tả, hoa/thường).
4. Trong project có `google_sheet_headers.csv`. Mở CSV bằng Notepad, copy **dòng duy nhất** gồm 34 tên cột.
5. Dán vào ô A1 của Sheet. Nếu tất cả nằm một ô, dùng **Data > Split text to columns** và chọn dấu phẩy. Cần 34 cột A–AH, thứ tự nguyên vẹn.

## 2. Apps Script

1. Trong Sheet chọn **Extensions > Apps Script**.
2. Xóa mã mặc định, dán **toàn bộ** file `Code.gs` từ project (mở bằng VS Code / Notepad), nhấn Save.
3. Chọn biểu tượng bánh răng **Project Settings > Script properties > Add script property**.
4. Tạo thuộc tính tên `INGEST_TOKEN` với giá trị secret dài ít nhất 24 ký tự. Tạo ngẫu nhiên bằng lệnh Windows:

   `.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"`

   Chép secret vào nơi riêng; **KHÔNG gửi cho người khác, không đăng GitHub**.
5. Trở lại Editor: **Deploy > New deployment > Select type > Web app**.
6. Chọn **Execute as: Me**, **Who has access: Anyone** (nếu Gmail cá nhân cho phép).
7. Chọn Deploy, cấp quyền cho script thao tác trên Sheet của chính bạn.
8. Copy Web app URL kết thúc bằng `/exec` (KHÔNG dùng `/dev`).

**Bảo mật:** Web app công khai để Python có thể POST, nhưng mỗi yêu cầu ghi cần `INGEST_TOKEN`. Đây là bảo vệ cơ bản dành cho demo, **không thích hợp cho dữ liệu bệnh nhân thật**. Nếu tài khoản trường chặn `Anyone`, thử Gmail cá nhân. Nếu Google hiển thị cảnh báo script chưa được xác minh, chỉ tiếp tục nếu bạn đã tự tạo và hiểu quyền được cấp.

## 3. Cấu hình Streamlit trên máy

1. Trong `diabetes_app/.streamlit/` copy `secrets.example.toml` rồi đổi tên bản copy thành `secrets.toml`.
2. Sửa hai giá trị sau (ví dụ):

   `apps_script_url = "https://script.google.com/macros/s/YOUR_ID/exec"`
   `logging_token = "YOUR_LONG_RANDOM_TOKEN"`

   Token phải **giống hệt** Project settings > Script properties > INGEST_TOKEN.
3. `secrets.toml` được `.gitignore` loại trừ khỏi GitHub; tự kiểm tra lại trước khi push.
4. Trong Terminal VS Code (đang đứng ở thư mục diabetes_app):

   `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
   `.\.venv\Scripts\python.exe test_logger_offline.py`
   `.\.venv\Scripts\python.exe -m streamlit run app.py`

5. Bấm Predict với dữ liệu mẫu: phải hiển thị 1.03%, có `prediction_id`, Google Sheet có dòng mới (34 cột); probability lưu = `0.010257108...` (KHÔNG phải `1.03`).
6. Thử thay token ở `secrets.toml` thành sai để xác nhận Predict vẫn ra kết quả nhưng có thông báo **NOT saved**. Sau đó sửa lại token đúng và khởi động lại Streamlit.

## 4. Khi deploy Streamlit Community Cloud

- Push file `app.py`, `logger.py`, `requirements.txt`, `deployment_bundle/` lên GitHub, KHÔNG push `.venv/`, `.streamlit/secrets.toml`, token hay các key.
- Trong app settings > Secrets của Streamlit Cloud, dán **hai dòng tương ứng** `apps_script_url` và `logging_token`.
- Chọn Python 3.13 nếu có. Deploy, Predict, và kiểm tra lại Sheet.
- Nếu có lỗi HTTP do giới hạn quyền Google hoặc redirect, kiểm tra đã chọn `Anyone`, đúng link `/exec`, và sau khi sửa Apps Script đã deploy phiên bản mới.

## 5. Giới hạn

- Apps Script có hạn mức sử dụng miễn phí phù hợp với demo nhỏ; đạt hạn mức thì sẽ bị từ chối/chậm, không tự phát sinh phí.
- Chưa gồm outcome update và retraining pipeline. Đây là công việc tiếp theo, không phải điều kiện để logging chạy.
- Ghi log bằng token không phải cơ chế bảo mật phù hợp với dữ liệu sức khỏe thật.
