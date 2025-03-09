# utils/utils.py
import torch

def cells_to_bboxes(predictions, anchors, strides, is_pred=True):
    """
    Chuyển đổi đầu ra của YOLOv5 từ grid cells sang bounding boxes.
    
    Args:
        predictions (list): Danh sách các tensor đầu ra từ YOLOHead [P3, P4, P5]
                            Shape: [batch_size, num_anchors, height, width, 5 + num_classes]
        anchors (list): Danh sách các anchor boxes đã được chia theo stride
                        Shape: 3 tensor [num_anchors_per_scale * 2]
        strides (list): Danh sách stride của từng detection layer [8, 16, 32]
        is_pred (bool): Nếu True, trả về dự đoán bounding box với class có xác suất cao nhất
                        Nếu False, giữ nguyên tất cả thông tin để tính loss
    
    Returns:
        bboxes (list): Danh sách các tensor bounding boxes cho từng scale
                       Shape mỗi tensor: [batch_size, num_anchors * height * width, 7]
                       7: [x, y, w, h, obj_score, class_id, class_score]
    """
    num_anchors_per_scale = len(anchors[0]) // 2
    bboxes = []

    for i, pred in enumerate(predictions):
        batch_size, num_anchors, height, width, _ = pred.shape
        stride = strides[i]
        anchor = anchors[i].reshape(1, num_anchors_per_scale, 1, 1, 2).to(pred.device)

        # Tạo grid tọa độ (Cx, Cy)
        x_grid = torch.arange(width, device=pred.device).repeat(height, 1).unsqueeze(-1)
        y_grid = torch.arange(height, device=pred.device).unsqueeze(-1).repeat(1, width).unsqueeze(-1)
        xy_grid = torch.cat([x_grid, y_grid], dim=-1).float()
        xy_grid = xy_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, num_anchors_per_scale, height, width, 2)

        # Chuyển đổi dự đoán
        pred_xy = (torch.sigmoid(pred[..., 0:2]) * 2 - 0.5 + xy_grid) * stride  # [x, y]
        pred_wh = ((torch.sigmoid(pred[..., 2:4]) * 2) ** 2) * anchor          # [w, h]
        pred_obj = torch.sigmoid(pred[..., 4:5])                                # objectness
        pred_cls = torch.sigmoid(pred[..., 5:])                                 # class probs

        # Kết hợp thành tensor
        bbox = torch.cat([pred_xy, pred_wh, pred_obj, pred_cls], dim=-1)

        if is_pred:
            # Chọn lớp có xác suất cao nhất
            cls_scores, cls_ids = pred_cls.max(dim=-1, keepdim=True)
            bbox = torch.cat([bbox[..., :5], cls_ids.float(), cls_scores], dim=-1)
            bbox = bbox.view(batch_size, -1, 7)  # [batch_size, num_boxes, 7]

        bboxes.append(bbox)

    return bboxes

