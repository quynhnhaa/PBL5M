# test_utils.py
import torch
import torchvision
from models.yolov5m import YOLOv5m
from utils.postprocess import cells_to_bboxes
from utils.evaluate import calculate_map
from config import Config

def test_utils():
    # Thiết lập device
    device = Config.DEVICE

    # Tạo dữ liệu giả lập
    batch_size = 2
    img_size = Config.IMG_SIZE
    num_scales = 3
    num_anchors = 3
    grid_sizes = [img_size // s for s in Config.STRIDES]  # [80, 40, 20]

    # Tạo ground truth giả lập (định dạng [class, x, y, w, h])
    true_bboxes = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1], [0, 0.3, 0.3, 0.05, 0.05]], device=device),  # Batch 0: 2 box
        torch.tensor([[0, 0.4, 0.4, 0.08, 0.08]], device=device)  # Batch 1: 1 box
    ]

    # Tạo preds giả lập
    preds = []
    for scale_idx, (stride, grid_size) in enumerate(zip(Config.STRIDES, grid_sizes)):
        batch_preds = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 5 + Config.NUM_CLASSES, device=device)
        
        # Thêm box giả lập vào scale 0 (stride=8) cho batch 0
        if scale_idx == 0:
            i, j = 40, 40  # Grid cell trung tâm (khoảng 320/8 = 40)
            batch_preds[0, 0, j, i, 0] = 0.5  # x_cell
            batch_preds[0, 0, j, i, 1] = 0.5  # y_cell
            batch_preds[0, 0, j, i, 2] = torch.log(torch.tensor(0.1 / Config.ANCHORS[0][0]))  # width
            batch_preds[0, 0, j, i, 3] = torch.log(torch.tensor(0.1 / Config.ANCHORS[0][1]))  # height
            batch_preds[0, 0, j, i, 4] = torch.log(torch.tensor(0.9))  # objectness (sigmoid ~ 0.71)
            batch_preds[0, 0, j, i, 5] = torch.log(torch.tensor(0.99))  # class 0 (sigmoid ~ 0.73)

        preds.append(batch_preds)

    # Test cells_to_bboxes
    print("Testing cells_to_bboxes...")
    pred_bboxes = cells_to_bboxes(
        preds,
        anchors=Config.ANCHORS,
        strides=Config.STRIDES,
        img_size=img_size,
        conf_thres=0.4,  # Giảm ngưỡng để giữ box
        iou_thres=0.5
    )
    
    # Kiểm tra output của cells_to_bboxes
    for batch_idx, boxes in enumerate(pred_bboxes):
        print(f"Batch {batch_idx}: {boxes.shape} boxes")
        if boxes.numel() > 0:
            print(f"Sample box: {boxes[0]}")  # In box đầu tiên (x1, y1, x2, y2, score, class)

    # Test calculate_map
    print("\nTesting calculate_map...")
    mAP = calculate_map(
        pred_bboxes=pred_bboxes,
        true_bboxes=true_bboxes,
        iou_thres=0.5,
        num_classes=Config.NUM_CLASSES
    )
    print(f"Final mAP: {mAP:.4f}")

if __name__ == "__main__":
    test_utils()