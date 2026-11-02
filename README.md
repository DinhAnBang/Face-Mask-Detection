# Face Mask Detection

Ứng dụng Tkinter kiểm thử model YOLO ONNX phát hiện khẩu trang trên khuôn mặt.

## Chức năng

- Chọn ảnh và vẽ bounding box cho từng khuôn mặt.
- Mở video hoặc webcam để phát hiện theo thời gian thực.
- Phân loại ba trạng thái: `with_mask`, `without_mask`, `mask_weared_incorrect`.
- Điều chỉnh ngưỡng confidence, áp dụng NMS và hiển thị số lượng từng loại.
- Hiển thị FPS và lưu ảnh đã vẽ kết quả.

## Cài đặt và chạy

Model mặc định được đọc từ `models/masked_face_model.onnx`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

`tkinter` thường được tích hợp sẵn trong bản Python trên Windows. Nếu cửa sổ không mở được, hãy kiểm tra lại bản cài Python có Tcl/Tk.

## Liên hệ với yêu cầu đề bài

| Yêu cầu | Phần triển khai |
| --- | --- |
| Object Detection | Model trả về bounding box, confidence và nhãn cho từng khuôn mặt. |
| Facial Analysis | Thống kê khuôn mặt có khẩu trang, không có khẩu trang và đeo sai. |
| Real-time Detection | Vòng lặp webcam/video cập nhật frame bằng Tkinter `after`. |
| Trực quan hóa kết quả | Bounding box màu, nhãn confidence, bảng thống kê và FPS. |
