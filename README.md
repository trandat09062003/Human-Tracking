# 🎯 Dự Án Theo Dõi Người (Human Tracking) - YOLOv11 & Roboflow

Dự án chuyên nghiệp sử dụng **YOLOv11** để nhận diện, theo dõi và đếm số lượng người từ video. Hệ thống được chia nhỏ thành các module chức năng để dễ dàng quản lý và triển khai.

---

## ✨ Tính Năng Nổi Bật

- **Tracking & Counting**: Theo dõi người với ID duy nhất và đếm số lượng trong thời gian thực.
- **Batch Processing**: Khả năng xử lý hàng loạt video tự động.
- **Auto-labeling Pipeline**: Tự động gán nhãn và upload lên Roboflow để mở rộng dữ liệu.
- **Validation**: Đánh giá độ chính xác của model trên tập dữ liệu kiểm thử.

---

## 🛠 Cấu Trúc Mã Nguồn & Cách Chạy

| File | Lệnh chạy | Chức năng |
| :--- | :--- | :--- |
| **`local_tracking.py`** | `python3 local_tracking.py` | Chạy tracking trên **1 video** cụ thể và hiển thị giao diện. |
| **`batch_processing.py`** | `python3 batch_processing.py` | Chạy tracking trên **tất cả video** trong thư mục và lưu vào `tracked_videos/`. |
| **`auto_label_and_upload.py`** | `python3 auto_label_and_upload.py` | Trích xuất ảnh từ video mới và đẩy lên Roboflow (Auto-label). |
| **`validate_model.py`** | `python3 validate_model.py` | Kiểm tra độ chính xác (mAP, Precision, Recall) của file `best.pt`. |
| **`best.pt`** | - | Trọng số model tốt nhất hiện tại. |

---

## 📋 Quy Trình Pipeline Đầy Đủ

1. **Thu thập dữ liệu**: Dùng `auto_label_and_upload.py` để lấy ảnh từ video thực tế.
2. **Gán nhãn**: Duyệt ảnh trên Roboflow (Assign & Annotate).
3. **Huấn luyện**: Upload dữ liệu lên Colab và train lại model (dùng `Human_Tracking_Training.ipynb`).
4. **Kiểm tra**: Dùng `validate_model.py` để xem model mới có tốt hơn model cũ không.
5. **Triển khai**: 
   - Chạy `local_tracking.py` để kiểm tra đơn lẻ.
   - Chạy `batch_processing.py` để xử lý số lượng lớn video.

---

## 💡 Lưu Ý Tùy Chỉnh

- **Thay đổi Video**: Sửa biến `VIDEO_SOURCE` trong các file tương ứng.
- **Đầu ra Video**: Mặc định lưu thành `[tên_file]_tracked.mp4`.
- **Cấu hình Roboflow**: Đảm bảo điền đúng API Key vào `auto_label_and_upload.py`.

---
*Phát triển bởi d@t - 2024*
