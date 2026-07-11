# Human Tracking

Nhận diện + theo dõi người trên video (CCTV / screen recording), giữ ID ổn định khi bị che khuất ngắn, và đếm người đi qua một đường ảo trên khung hình.

Model hiện dùng: `best_v5.pt` (YOLO11 fine-tune 1 class `person`). Tracker: BoT-SORT + lớp `IDRecoverer` tự viết phía trên.

![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ftrandat09062003%2FHuman-Tracking&count_bg=%2379C0FF&title_bg=%23555555&icon=&icon_color=%23E5E5E5&title=views&edge_flat=false)

---

## Làm được gì

- Detect người mỗi frame bằng YOLO11 (`best_v5.pt`)
- Theo dõi multi-object, mỗi người một `stable_id` (không dùng thẳng raw ID của BoT-SORT)
- Lọc false positive trước khi đưa vào tracker (xe đạp / xe máy hay bị nhận nhầm người khi conf thấp)
- Giảm ID switch / ID mới oan khi che khuất vài giây
- Vẽ box + nhãn ID + HUD (People / Unique / RTL / FPS)
- Đếm người cắt đường đỏ, mặc định chỉ tính hướng phải → trái (RTL)
- Xuất video `*_tracked.mp4`

Chi tiết tham số, trade-off và lý do chọn từng lớp xử lý nằm trong [`summary.txt`](summary.txt).

---

## Pipeline (không dùng `model.track()` end-to-end)

```
Frame
  → YOLO.predict (best_v5.pt, class person)
  → Pre-filter (conf / aspect / area / nested / overlap)
  → BoT-SORT (custom_tracker.yaml)  → raw track ID
  → Prune raw tracks chồng nhau
  → IDRecoverer (Hungarian + HSV torso + velocity)  → stable_id
  → Suppress 2 stable ID chồng → giữ ID già hơn
  → LineCounter (điểm chân cắt đường đỏ, only_rtl)
  → Vẽ + ghi video
```

Lý do tách detect và track thủ công:

1. Phải lọc FP / box chồng **trước** khi tracker cấp ID. Lọc sau `model.track()` thì ID rác đã được tạo rồi.
2. Cần lớp ổn định ID riêng (`IDRecoverer`) vì BoT-SORT vẫn hay cấp raw ID mới sau miss ngắn.
3. Kiểm soát được demote / alias khi hai box cùng một người.

---

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

Cần: `ultralytics`, `opencv-python`, `torch`, `numpy`, `scipy`, `pyyaml`.  
Có GPU thì PyTorch dùng CUDA; không có thì chạy CPU (chậm hơn nhưng ổn định ID vẫn được ưu tiên hơn FPS realtime).

---

## Cấu trúc repo

```text
Human_Tracking/
├── local_tracking.py              # pipeline chính
├── custom_tracker.yaml            # cấu hình BoT-SORT
├── best_v5.pt                     # model person (fine-tune)
├── best.pt                        # bản cũ (tham khảo)
├── requirements.txt
├── summary.txt                    # ghi chú kỹ thuật / tham số
├── auto_label_and_upload.py       # cắt frame + nhãn YOLO → Roboflow
├── Human_Tracking_Training.ipynb  # train lại trên Colab
├── Data/                          # video đầu vào (local, không commit)
└── README.md
```

Dataset thô (`abc/`, `abc_autolabel/`, video `.mp4`) nằm trong `.gitignore`.

---

## Chạy tracking

1. Mở `local_tracking.py`, sửa đường dẫn video:

```python
VIDEO_SOURCE = r"C:\path\to\video.mp4"
MODEL_PATH = "best_v5.pt"
TRACKER_YAML = "custom_tracker.yaml"
```

Trên Windows nhớ dùng raw string (`r"..."`) để tránh lỗi `\U` trong đường dẫn.

2. Chạy:

```bash
python local_tracking.py
```

3. Cửa sổ preview: nhấn `q` để dừng. Video kết quả ghi cạnh file nguồn, tên `*_tracked.mp4`.

HUD góc trên trái:

| Dòng   | Ý nghĩa                                      |
|--------|----------------------------------------------|
| People | Số người đang hiện trên frame                |
| Unique | Tổng `stable_id` đã xuất hiện trong clip     |
| RTL    | Số người đã cắt đường theo hướng phải→trái   |
| FPS    | Tốc độ xử lý                                 |

---

## Đếm người qua đường (LineCounter)

Đường đỏ cấu hình bằng tọa độ chuẩn hóa 0–1:

```python
COUNT_LINE_NORM = ((0.18, 0.58), (0.92, 0.72))
```

- Điểm theo dõi mặc định: **chân** (đáy giữa bounding box), không dùng tâm box — giảm đếm nhầm khi nửa người còn trên đường.
- Coi là cắt đường khi đoạn chân frame trước → frame này giao đoạn đường, **hoặc** đổi phía so với đường (dấu tích có hướng 2D đổi dấu).
- `only_rtl=True`: chỉ cộng khi `dx <= 0` (x giảm = đi phải → trái).
- Mỗi `stable_id` chỉ đếm **một lần** (`counted_ids`).

Chỉnh `COUNT_LINE_NORM` cho khớp camera / góc quay. Hai đầu đường vẽ chấm đỏ trên preview để dễ canh.

---

## Tham số hay chỉnh

### Trong `local_tracking.py`

| Tham số | Hiện tại | Ghi chú ngắn |
|---------|----------|--------------|
| `DETECT_CONF` | 0.28 | Lấy candidate rộng |
| `TRACK_CONF` | 0.42 | Chỉ box ≥ ngưỡng này vào tracker (chặn FP xe đạp ~0.39) |
| `IOU` | 0.28 | NMS YOLO chặt, ít double-box |
| `MIN_ASPECT` / `MAX_ASPECT` | 1.70 / 7.00 | Người đứng CCTV thường cao hơn rộng |
| `NEST_THRESH` | 0.40 | Box nhỏ nằm trong box lớn → bỏ |
| `OVERLAP_IOU` | 0.25 | Hai box chồng mạnh → giữ conf cao hơn |
| `TRACK_OVERLAP_IOU` | 0.28 | Hai stable ID chồng → giữ ID già hơn |
| `MIN_HITS` | 2 | Hiện track sau 2 frame khớp |
| `RECOVER_MAX_FRAMES` | 150 | Nhớ lost ID ~5s @ 30fps |

### Trong `custom_tracker.yaml`

| Tham số | Hiện tại | Ghi chú ngắn |
|---------|----------|--------------|
| `tracker_type` | botsort | |
| `track_buffer` | 150 | Giữ lost ~5s |
| `match_thresh` | 0.80 | Dễ khớp lại sau miss (cost ≈ 1−IoU) |
| `new_track_thresh` | 0.45 | Ngưỡng tạo raw ID mới |
| `gmc_method` | none | Camera cố định, không bật GMC |
| `with_reid` | False | Không dùng deep ReID (nặng trên CPU) |

Bảng “tăng/giảm tham số thì sao” xem `summary.txt` mục 5.

---

## IDRecoverer (tóm tắt)

BoT-SORT trả về `raw_id`. `IDRecoverer` map sang `stable_id` và cố gắng gắn lại ID cũ sau khi mất track:

- Ghép Hungarian (`scipy.optimize.linear_sum_assignment`)
- Cost: khoảng cách tới vị trí dự đoán (velocity) + histogram HSV vùng thân + IoU + tỉ lệ diện tích
- Chặn recover khi detection giống người đang active cạnh hơn là track đang mất (tránh cướp ID)
- Cắt liên kết raw→stable khi nhảy xa **và** appearance khác rõ (teleport)

---

## Thu thập data / train lại model

1. Chạy `auto_label_and_upload.py` để cắt frame có người, tạo nhãn YOLO, upload Roboflow (cần API key / project của bạn).
2. Trên Roboflow: review nhãn → Generate New Version.
3. Mở `Human_Tracking_Training.ipynb` trên Colab, train YOLO11, tải `best.pt` / đổi tên thành `best_v5.pt` nếu muốn thay model hiện tại.
4. Chạy lại `local_tracking.py` để kiểm tra.

---

## Lỗi thường gặp

**`unicodeescape` trên Windows**  
Dùng raw string: `VIDEO_SOURCE = r"C:\...\video.mp4"`.

**`AttributeError: ... no attribute 'fuse_score'`**  
File yaml tracker thiếu field so với bản Ultralytics đang cài. Dùng nguyên `custom_tracker.yaml` trong repo (đã khai báo đủ).

**Mở không được `*_tracked.mp4`**  
Phải thoát bằng `q` để OpenCV đóng writer đúng cách. Kill process giữa chừng dễ ra file hỏng.

**Đếm RTL = 0 dù thấy người qua đường**  
Canh lại `COUNT_LINE_NORM`; kiểm tra hướng đi (mặc định chỉ RTL); xem ID có ổn định không (ID nhảy liên tục thì `counted_ids` / điểm chân dễ lệch).

**Nhiều Unique ID ảo**  
Thường do FP vào tracker hoặc recover quá chặt / buffer ngắn. Tăng `TRACK_CONF`, siết overlap, hoặc nới `RECOVER_*` — xem trade-off trong `summary.txt`.

---

## Ghi chú

- Ưu tiên ổn định ID trên CPU hơn là realtime tuyệt đối.
- `yolo11n.pt` / video test local không cần commit.
- Không commit API key Roboflow lên GitHub.
