import os
import torch

CHECKPOINT_DIR = "/kaggle/working/weights"
CHECKPOINT_DIR = "weights"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

class Config:
    # BASE_DIR = "/kaggle/input/datasetmy/data/data"
    BASE_DIR = "data"

    IMG_SIZE = 640
    NUM_CLASSES = 1
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ANCHORS = [
        [0.0102, 0.0178, 0.0257, 0.0466, 0.0510, 0.0887],
        [0.0938, 0.1536, 0.1303, 0.2707, 0.2502, 0.2486],
        [0.2246, 0.4716, 0.4321, 0.4260, 0.4500, 0.7483]
    ]
    STRIDES = [8, 16, 32]

    TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images/train")
    TRAIN_LABEL_DIR = os.path.join(BASE_DIR, "labels/train")
    VAL_IMG_DIR = os.path.join(BASE_DIR, "images/val")
    VAL_LABEL_DIR = os.path.join(BASE_DIR, "labels/val")

    CHECKPOINT_DIR = CHECKPOINT_DIR
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "yolov5m_epoch_{epoch}.pth")
