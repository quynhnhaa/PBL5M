# config.py
import os
import torch

CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

class Config:
    IMG_SIZE = 640
    NUM_CLASSES = 1
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0001  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ANCHORS = [
        [1.25, 1.625, 2.0, 3.75, 4.125, 2.875],
        [1.875, 3.8125, 4.0, 2.0, 3.75, 5.0],
        [4.0, 7.0, 8.0, 4.0, 6.0, 10.0]
    ]
    STRIDES = [8, 16, 32]
    TRAIN_IMG_DIR = "data/images/train"
    TRAIN_LABEL_DIR = "data/labels/train"
    VAL_IMG_DIR = "data/images/val"
    VAL_LABEL_DIR = "data/labels/val"
    CHECKPOINT_DIR = CHECKPOINT_DIR
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "yolov5m_epoch_{epoch}.pth")