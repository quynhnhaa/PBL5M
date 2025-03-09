# config.py
import os
import torch

# Thư mục lưu checkpoint
CHECKPOINT_DIR = "/kaggle/working/weights"  # Đường dẫn trên Kaggle
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

class Config:
    # Đường dẫn dữ liệu (sẽ được cập nhật trong Notebook)
    BASE_DIR = None  # Sẽ gán giá trị từ Notebook

    IMG_SIZE = 640
    NUM_CLASSES = 1  # Thay bằng số lớp thực tế của dataset
    BATCH_SIZE = 16  # Giảm từ 32 để tránh lỗi bộ nhớ trên Kaggle
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Anchor và stride (giữ nguyên)
    ANCHORS = [
        [1.25, 1.625, 2.0, 3.75, 4.125, 2.875],
        [1.875, 3.8125, 4.0, 2.0, 3.75, 5.0],
        [4.0, 7.0, 8.0, 4.0, 6.0, 10.0]
    ]
    STRIDES = [8, 16, 32]

    # Đường dẫn ảnh và label (sẽ được cập nhật dựa trên BASE_DIR)
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images/train")
    TRAIN_LABEL_DIR = os.path.join(BASE_DIR, "labels/train")
    VAL_IMG_DIR = os.path.join(BASE_DIR, "images/val")
    VAL_LABEL_DIR = os.path.join(BASE_DIR, "labels/val")

    CHECKPOINT_DIR = CHECKPOINT_DIR
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "yolov5m_epoch_{epoch}.pth")