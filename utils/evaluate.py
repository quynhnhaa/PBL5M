import torch
import torchvision
from config import Config
def calculate_map(pred_bboxes, true_bboxes, iou_thres=0.5, num_classes=1):
    """
    Tính mAP trên tập validation.

    Args:
        pred_bboxes (list): Dự đoán từ cells_to_bboxes, [batch_size] của tensor [num_boxes, 6].
        true_bboxes (list): Ground truth, [batch_size] của tensor [num_boxes, 5] (class, x, y, w, h).
        iou_thres (float): Ngưỡng IoU để xác định True Positive.
        num_classes (int): Số class (Config.NUM_CLASSES).

    Returns:
        float: mAP@iou_thres.
    """
    # Khởi tạo danh sách để lưu AP cho từng class
    average_precisions = []
    
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Gộp tất cả dự đoán và ground truth cho class c
        for batch_idx in range(len(pred_bboxes)):
            # Dự đoán
            pred_boxes = pred_bboxes[batch_idx]
            if pred_boxes.numel() > 0:
                pred_class_mask = pred_boxes[..., 5] == c
                pred_boxes_c = pred_boxes[pred_class_mask]
                for box in pred_boxes_c:
                    detections.append([batch_idx, box[4].item(), *box[:4].tolist()])

            # Ground truth
            true_boxes = true_bboxes[batch_idx]
            if true_boxes.numel() > 0:
                true_class_mask = true_boxes[..., 0] == c
                true_boxes_c = true_boxes[true_class_mask]
                for box in true_boxes_c:
                    ground_truths.append([batch_idx, *box[1:].tolist()])

        # Nếu không có ground truth cho class này, bỏ qua
        if len(ground_truths) == 0:
            continue

        # Tạo danh sách nhãn (1: True Positive, 0: False Positive)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)  # Sắp xếp theo score
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # Tính IoU giữa dự đoán và ground truth
        for det_idx, detection in enumerate(detections):
            batch_idx = detection[0]
            gt_bboxes = [gt[1:] for gt in ground_truths if gt[0] == batch_idx]

            if len(gt_bboxes) == 0:
                FP[det_idx] = 1
                continue

            gt_bboxes = torch.tensor(gt_bboxes, device=Config.DEVICE)
            pred_box = torch.tensor(detection[2:6], device=Config.DEVICE).unsqueeze(0)

            # Chuyển ground truth thành x1, y1, x2, y2
            gt_boxes_xyxy = torch.zeros_like(gt_bboxes)
            gt_boxes_xyxy[..., 0] = gt_bboxes[..., 0] * Config.IMG_SIZE - gt_bboxes[..., 2] * Config.IMG_SIZE / 2
            gt_boxes_xyxy[..., 1] = gt_bboxes[..., 1] * Config.IMG_SIZE - gt_bboxes[..., 3] * Config.IMG_SIZE / 2
            gt_boxes_xyxy[..., 2] = gt_bboxes[..., 0] * Config.IMG_SIZE + gt_bboxes[..., 2] * Config.IMG_SIZE / 2
            gt_boxes_xyxy[..., 3] = gt_bboxes[..., 1] * Config.IMG_SIZE + gt_bboxes[..., 3] * Config.IMG_SIZE / 2

            ious = torchvision.ops.box_iou(pred_box, gt_boxes_xyxy)
            max_iou, max_idx = ious.max(dim=1)

            if max_iou > iou_thres:
                # Kiểm tra xem ground truth này đã được gán chưa
                if all(d[0] != batch_idx or not torch.equal(torch.tensor(d[2:6], device=Config.DEVICE), gt_bboxes[max_idx]) for d in detections[:det_idx]):
                    TP[det_idx] = 1
                else:
                    FP[det_idx] = 1
            else:
                FP[det_idx] = 1

        # Tính Precision và Recall
        cum_TP = torch.cumsum(TP, dim=0)
        cum_FP = torch.cumsum(FP, dim=0)
        precision = cum_TP / (cum_TP + cum_FP + 1e-16)
        recall = cum_TP / total_true_bboxes

        # Tính AP bằng phương pháp 11-point interpolation
        recall = torch.cat([torch.tensor([0.0], device=Config.DEVICE), recall, torch.tensor([1.0], device=Config.DEVICE)])
        precision = torch.cat([torch.tensor([0.0], device=Config.DEVICE), precision, torch.tensor([0.0], device=Config.DEVICE)])

        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = recall >= t
            p = precision[mask].max() if mask.any() else 0.0
            ap += p / 11

        average_precisions.append(ap)

    mAP = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    print(f"mAP@{iou_thres:.1f}: {mAP:.4f}")
    return mAP