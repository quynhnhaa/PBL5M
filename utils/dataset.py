# utils/datasets.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, anchors, img_size=640, transform=None):
        """
        Khởi tạo dataset cho YOLOv5.
        
        Args:
            img_dir (str): Đường dẫn đến thư mục chứa ảnh (images/train hoặc images/val)
            label_dir (str): Đường dẫn đến thư mục chứa nhãn (labels/train hoặc labels/val)
            anchors (list): Danh sách anchor boxes
            img_size (int): Kích thước ảnh đầu vào (mặc định 640)
            transform (callable, optional): Hàm tăng cường dữ liệu
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.anchors = torch.tensor(anchors).float()
        self.strides = torch.tensor([8, 16, 32])

        # Lấy danh sách tất cả các file ảnh từ các thư mục con
        self.img_files = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.img_files.append(os.path.join(root, file))

    def __len__(self):
        """Trả về số lượng ảnh trong dataset."""
        return len(self.img_files)

    def __getitem__(self, idx):
        """Lấy một mẫu dữ liệu (ảnh và nhãn)."""
        # Đường dẫn ảnh
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img) / 255.0
        img = torch.FloatTensor(img).permute(2, 0, 1)  # [C, H, W]

        # Đường dẫn nhãn (thay .jpg bằng .txt và điều chỉnh thư mục)
        label_path = img_path.replace(self.img_dir, self.label_dir).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        boxes = []

        # Đọc nhãn (định dạng YOLO: [class_id, x_center, y_center, width, height])
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                # Đọc nhãn (giả định định dạng WIDERFACE: x1, y1, w, h, chuyển sang YOLO: class_id, x_center, y_center, w, h)
                # for line in lines:
                #     parts = line.strip().split()
                #     if len(parts) >= 4:  # Đảm bảo có đủ tọa độ
                #         x1, y1, w, h = map(float, parts[:4])  # Giả định WIDERFACE format
                #         # Chuyển đổi sang YOLO format
                #         img_w, img_h = self.img_size, self.img_size
                #         x_center = (x1 + w / 2) / img_w
                #         y_center = (y1 + h / 2) / img_h
                #         width = w / img_w
                #         height = h / img_h
                #         # Giả định class_id = 0 (mặc định cho WIDERFACE là face detection)
                #         boxes.append([0, x_center, y_center, width, height])

                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:  # Đảm bảo có đủ 5 giá trị
                        class_id, x_center, y_center, width, height = parts
                        boxes.append([class_id, x_center, y_center, width, height])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Áp dụng transform nếu có
        if self.transform:
            img, boxes = self.transform(img, boxes)

        return img, boxes

# Hàm collate để xử lý batch (gộp các boxes thành list)
def collate_fn(batch):
    imgs, boxes = zip(*batch)
    return torch.stack(imgs), boxes


# if __name__ == "__main__":
#     anchors = [
#         [1.25, 1.625, 2.0, 3.75, 4.125, 2.875],  # P3
#         [1.875, 3.8125, 4.0, 2.0, 3.75, 5.0],   # P4
#         [4.0, 7.0, 8.0, 4.0, 6.0, 10.0]         # P5
#     ]
    
#     dataset = YOLODataset(
#         img_dir="data/images/train",
#         label_dir="data/labels/train",
#         anchors=anchors,
#         img_size=640
#     )
    
#     loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
#     for imgs, targets in loader:
#         print(f"Batch images shape: {imgs.shape}")  # [batch_size, 3, 640, 640]
#         print(f"Number of targets: {len(targets)}")  # [batch_size] list of tensors
#         for target in targets:
#             print(f"Target shape: {target.shape if target.numel() > 0 else 'Empty'}")  # [num_boxes, 5]
#         break