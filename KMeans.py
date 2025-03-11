import os
import numpy as np
from sklearn.cluster import KMeans
from config import Config

# Hàm trích xuất kích thước bounding box từ file label, duyệt qua các thư mục con
def extract_boxes(label_dir):
    boxes = []
    # Duyệt qua tất cả các thư mục và file trong label_dir
    for root, dirs, files in os.walk(label_dir):
        for file_name in files:
            if file_name.endswith('.txt'):  # Chỉ xử lý file .txt
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:  # Đảm bảo định dạng đúng (class, x_center, y_center, width, height)
                            _, _, _, width, height = map(float, parts)
                            # Chuyển width, height chuẩn hóa về pixel (IMG_SIZE)
                            box_width = width * Config.IMG_SIZE
                            box_height = height * Config.IMG_SIZE
                            boxes.append([box_width, box_height])
    return np.array(boxes)

# Thư mục chứa label
train_label_dir = Config.TRAIN_LABEL_DIR
val_label_dir = Config.VAL_LABEL_DIR

# Kết hợp tất cả bounding box từ train và val
all_boxes = np.vstack([
    extract_boxes(train_label_dir),
    extract_boxes(val_label_dir)
])

# Kiểm tra nếu có đủ bounding box
if len(all_boxes) == 0:
    raise ValueError("Không tìm thấy bounding box trong dữ liệu label. Kiểm tra lại thư mục label hoặc định dạng file!")

# Thực hiện k-means clustering với 9 cụm (9 anchor)
kmeans = KMeans(n_clusters=9, random_state=42).fit(all_boxes)
anchors = kmeans.cluster_centers_

# Sắp xếp anchor theo diện tích (width * height) để gán cho các scale
anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
anchors = anchors.reshape(3, 3, 2)  # 3 scale, mỗi scale 3 anchor

# Chuẩn hóa anchor về [0, 1] dựa trên IMG_SIZE
anchors_normalized = anchors / Config.IMG_SIZE

# In kết quả
print("Anchor mới (theo pixel):")
print(anchors)
print("\nAnchor mới (chuẩn hóa [0, 1]):")
print(anchors_normalized)