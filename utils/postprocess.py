# utils/postprocess.py
import torch
import torchvision

def cells_to_bboxes(preds, anchors, strides, img_size, conf_thres=0.5, iou_thres=0.5):
    """
    Chuyển đổi dự đoán từ mô hình YOLOv5m thành bounding box.

    Args:
        preds (list): List 3 tensor từ YOLOv5m, mỗi tensor có shape [batch_size, num_anchors, grid_h, grid_w, 5+num_classes].
        anchors (list): Anchors từ Config.ANCHORS, định dạng [num_scales, num_anchors * 2].
        strides (list): Strides từ Config.STRIDES.
        img_size (int): Kích thước ảnh (Config.IMG_SIZE).
        conf_thres (float): Ngưỡng confidence để lọc box.
        iou_thres (float): Ngưỡng IoU để áp dụng NMS.

    Returns:
        list: Danh sách bounding box sau khi NMS, mỗi phần tử là tensor [batch_size, num_boxes, 6] (x1, y1, x2, y2, conf, class).
    """
    batch_size = preds[0].shape[0]
    num_classes = preds[0].shape[-1] - 5
    num_scales = len(anchors)
    num_anchors = len(anchors[0]) // 2  # 3 anchors mỗi scale

    # Chuyển anchors từ list 2D thành tensor 3D [num_scales, num_anchors, 2]
    anchors_tensor = torch.tensor(anchors, device=preds[0].device).view(num_scales, num_anchors, 2)

    all_bboxes = []

    for scale_idx, pred in enumerate(preds):
        # Reshape và tính tọa độ
        num_anchors_scale, grid_h, grid_w = pred.shape[1:4]
        pred = pred.reshape(batch_size, num_anchors_scale * grid_h * grid_w, 5 + num_classes)

        # Tính tọa độ x, y, w, h
        grid_x = torch.arange(grid_w, device=pred.device).repeat(grid_h, 1).view(1, 1, grid_h, grid_w)
        grid_y = torch.arange(grid_h, device=pred.device).repeat_interleave(grid_w).view(1, 1, grid_h, grid_w)
        grid_x = grid_x.repeat(batch_size, num_anchors_scale, 1, 1).reshape(batch_size, num_anchors_scale * grid_h * grid_w)
        grid_y = grid_y.repeat(batch_size, num_anchors_scale, 1, 1).reshape(batch_size, num_anchors_scale * grid_h * grid_w)

        anchor_w = anchors_tensor[scale_idx, :, 0].view(1, num_anchors_scale, 1, 1).repeat(batch_size, 1, grid_h, grid_w).reshape(batch_size, num_anchors_scale * grid_h * grid_w)
        anchor_h = anchors_tensor[scale_idx, :, 1].view(1, num_anchors_scale, 1, 1).repeat(batch_size, 1, grid_h, grid_w).reshape(batch_size, num_anchors_scale * grid_h * grid_w)

        pred_boxes = torch.zeros_like(pred[..., :4])
        pred_boxes[..., 0] = (torch.sigmoid(pred[..., 0]) + grid_x) * strides[scale_idx]  # x_center
        pred_boxes[..., 1] = (torch.sigmoid(pred[..., 1]) + grid_y) * strides[scale_idx]  # y_center
        pred_boxes[..., 2] = torch.exp(pred[..., 2]) * anchor_w  # width
        pred_boxes[..., 3] = torch.exp(pred[..., 3]) * anchor_h  # height

        # Chuyển thành x1, y1, x2, y2
        bboxes = torch.zeros_like(pred_boxes)
        bboxes[..., 0] = pred_boxes[..., 0] - pred_boxes[..., 2] / 2  # x1
        bboxes[..., 1] = pred_boxes[..., 1] - pred_boxes[..., 3] / 2  # y1
        bboxes[..., 2] = pred_boxes[..., 0] + pred_boxes[..., 2] / 2  # x2
        bboxes[..., 3] = pred_boxes[..., 1] + pred_boxes[..., 3] / 2  # y2

        # Lấy confidence và class
        conf = torch.sigmoid(pred[..., 4])
        class_probs = torch.sigmoid(pred[..., 5:])
        class_scores, class_ids = class_probs.max(dim=-1)

        # Tính score cuối (conf * class_prob)
        scores = conf * class_scores

        # Gộp tất cả thành [batch_size, num_boxes, 6] (x1, y1, x2, y2, score, class)
        boxes = torch.cat([bboxes, scores.unsqueeze(-1), class_ids.unsqueeze(-1)], dim=-1)
        all_bboxes.append(boxes)

    # Gộp các scale và áp dụng NMS
    all_bboxes = torch.cat(all_bboxes, dim=1)  # [batch_size, total_boxes, 6]

    final_bboxes = []
    for b in range(batch_size):
        boxes = all_bboxes[b]
        scores = boxes[..., 4]
        classes = boxes[..., 5].long()

        # Lọc box theo confidence
        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # Áp dụng NMS
        indices = torchvision.ops.nms(boxes[..., :4], scores, iou_thres)
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]

        # Gộp lại thành [num_boxes, 6]
        final = torch.cat([boxes[..., :4], scores.unsqueeze(-1), classes.unsqueeze(-1)], dim=-1)
        final_bboxes.append(final)

    return final_bboxes