import torch
import torch.nn as nn
import os
import csv
import math

def bbox_iou(box1, box2, x1y1x2y2=False, CIoU=True):
    """Tính CIoU giữa hai bounding box."""
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter = inter_w * inter_h

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + (w2 * h2) - inter

    iou = inter / union

    if CIoU:
        center_distance = torch.pow(b2_x1 + b2_x2 - b1_x1 - b1_x2, 2) + torch.pow(b2_y1 + b2_y2 - b1_y1 - b1_y2, 2)
        diag_distance = torch.pow(b2_x2 - b2_x1, 2) + torch.pow(b2_y2 - b2_y1, 2) + 1e-16
        u = center_distance / diag_distance

        with torch.no_grad():
            arctan = torch.atan(w2 / (h2 + 1e-16)) - torch.atan(w1 / (h1 + 1e-16))
            v = 4 / (math.pi ** 2) * torch.pow(arctan, 2)
            s = 1 - iou
            alpha = v / (s + v + 1e-16)

        return iou - u - alpha * v
    return iou

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=1, anchors=None, strides=[8, 16, 32], device="cuda", save_logs=False, filename=None):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors, dtype=torch.float32).view(3, -1, 2).to(device) if anchors else None
        self.strides = torch.tensor(strides, dtype=torch.float32).to(device)
        self.device = device
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="sum")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="sum")
        self.lambda_coord = 0.05 * (3 / len(strides))
        self.lambda_obj = 1.0 * ((640 / 640) ** 2 * 3 / len(strides))
        self.lambda_noobj = 0.1
        self.lambda_cls = 0.5 * (num_classes / 80 * 3 / len(strides))
        self.balance = [4.0, 1.0, 0.4]
        self.save_logs = save_logs
        self.filename = filename

        if save_logs:
            folder = os.path.join("train_eval_metrics", filename)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            with open(os.path.join(folder, "loss.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "batch_idx", "coord_loss", "obj_loss", "noobj_loss", "cls_loss"])
                print(f"Training Logs will be saved in {os.path.join('train_eval_metrics', filename, 'loss.csv')}")
                f.close()

    def build_targets(self, preds, targets):
        targets_per_scale = [torch.zeros(pred.shape, dtype=torch.float32, device=self.device) for pred in preds]
        anchors_per_scale = self.anchors

        for batch_idx, target in enumerate(targets):
            if target.numel() == 0:
                continue

            target_boxes = target[:, 1:] * self.img_size
            classes = target[:, 0].long()

            for idx, box in enumerate(target_boxes):
                x, y, w, h = box
                best_iou = 0
                best_scale_idx = 0
                best_anchor_idx = 0

                for scale_idx, anchors in enumerate(anchors_per_scale):
                    iou_anchors = self.iou_width_height(anchors, w, h)
                    max_iou, max_idx = iou_anchors.max(dim=0)
                    if max_iou > best_iou:
                        best_iou = max_iou
                        best_scale_idx = scale_idx
                        best_anchor_idx = max_idx

                if best_iou < 0.3:
                    continue

                grid_size = self.img_size // self.strides[best_scale_idx]
                i, j = int(x // self.strides[best_scale_idx]), int(y // self.strides[best_scale_idx])

                if i >= grid_size or j >= grid_size or i < 0 or j < 0:
                    continue

                x_cell = (x / self.strides[best_scale_idx]) - j
                y_cell = (y / self.strides[best_scale_idx]) - i
                width_cell = torch.log(w / anchors_per_scale[best_scale_idx, best_anchor_idx, 0] + 1e-16)
                height_cell = torch.log(h / anchors_per_scale[best_scale_idx, best_anchor_idx, 1] + 1e-16)

                targets_per_scale[best_scale_idx][batch_idx, best_anchor_idx, j, i, 0] = x_cell
                targets_per_scale[best_scale_idx][batch_idx, best_anchor_idx, j, i, 1] = y_cell
                targets_per_scale[best_scale_idx][batch_idx, best_anchor_idx, j, i, 2] = width_cell
                targets_per_scale[best_scale_idx][batch_idx, best_anchor_idx, j, i, 3] = height_cell
                targets_per_scale[best_scale_idx][batch_idx, best_anchor_idx, j, i, 4] = 1.0
                targets_per_scale[best_scale_idx][batch_idx, best_anchor_idx, j, i, 5] = classes[idx]

                for other_scale_idx, other_anchors in enumerate(anchors_per_scale):
                    if other_scale_idx == best_scale_idx:
                        continue
                    other_iou = self.iou_width_height(other_anchors, w, h).max()
                    if other_iou > 0.5:
                        other_grid_size = self.img_size // self.strides[other_scale_idx]
                        other_i, other_j = int(x // self.strides[other_scale_idx]), int(y // self.strides[other_scale_idx])
                        if (0 <= other_i < other_grid_size) and (0 <= other_j < other_grid_size):
                            for a in range(anchors_per_scale.shape[1]):
                                targets_per_scale[other_scale_idx][batch_idx, a, other_j, other_i, 4] = -1

        return targets_per_scale

    def iou_width_height(self, anchors, w, h):
        w_anchors = anchors[..., 0]
        h_anchors = anchors[..., 1]
        w_inter = torch.min(w, w_anchors)
        h_inter = torch.min(h, h_anchors)
        inter = w_inter * h_inter
        union = (w * h) + (w_anchors * h_anchors) - inter
        return inter / (union + 1e-16)

    def forward(self, preds, targets, img_size=640, batch_idx=None, epoch=None):
        self.img_size = img_size
        targets_per_scale = self.build_targets(preds, targets)

        total_loss = 0
        coord_loss_total = 0
        obj_loss_total = 0
        noobj_loss_total = 0
        cls_loss_total = 0

        for i, (pred, target) in enumerate(zip(preds, targets_per_scale)):
            batch_size = pred.shape[0]
            num_anchors, grid_h, grid_w = pred.shape[1:4]
            pred = pred.reshape(batch_size, num_anchors * grid_h * grid_w, -1)
            target = target.reshape(batch_size, num_anchors * grid_h * grid_w, -1)

            obj_mask = target[..., 4] == 1
            noobj_mask = target[..., 4] == 0
            ignore_mask = target[..., 4] == -1

            # Tính CIoU Loss cho tọa độ
            pred_boxes = torch.zeros_like(pred[..., :4])
            grid_x = torch.arange(grid_w, device=self.device).repeat(grid_h, 1).view(1, 1, grid_h, grid_w)
            grid_y = torch.arange(grid_h, device=self.device).repeat_interleave(grid_w).view(1, 1, grid_h, grid_w)
            grid_x = grid_x.repeat(batch_size, num_anchors, 1, 1).reshape(batch_size, num_anchors * grid_h * grid_w)
            grid_y = grid_y.repeat(batch_size, num_anchors, 1, 1).reshape(batch_size, num_anchors * grid_h * grid_w)
            anchor_w = self.anchors[i][:, 0].view(1, num_anchors, 1, 1).repeat(batch_size, 1, grid_h, grid_w).reshape(batch_size, num_anchors * grid_h * grid_w)
            anchor_h = self.anchors[i][:, 1].view(1, num_anchors, 1, 1).repeat(batch_size, 1, grid_h, grid_w).reshape(batch_size, num_anchors * grid_h * grid_w)

            pred_boxes[..., 0] = (torch.sigmoid(pred[..., 0]) + grid_x) / grid_w
            pred_boxes[..., 1] = (torch.sigmoid(pred[..., 1]) + grid_y) / grid_h
            pred_boxes[..., 2] = torch.exp(pred[..., 2]) * anchor_w
            pred_boxes[..., 3] = torch.exp(pred[..., 3]) * anchor_h

            target_boxes = torch.zeros_like(target[..., :4])
            target_boxes[..., 0] = (target[..., 0] + grid_x) / grid_w
            target_boxes[..., 1] = (target[..., 1] + grid_y) / grid_h
            target_boxes[..., 2] = torch.exp(target[..., 2]) * anchor_w
            target_boxes[..., 3] = torch.exp(target[..., 3]) * anchor_h

            coord_mask = obj_mask.unsqueeze(-1).expand_as(target[..., :4])
            pred_boxes = pred_boxes[coord_mask].reshape(-1, 4)
            target_boxes = target_boxes[coord_mask].reshape(-1, 4)
            coord_loss = -bbox_iou(pred_boxes, target_boxes, x1y1x2y2=False, CIoU=True) if pred_boxes.numel() > 0 else torch.tensor(0.0).to(self.device)
            coord_loss = coord_loss.sum() * self.lambda_coord * self.balance[i]

            # Obj/NoObj Loss với điều chỉnh IoU
            iou = bbox_iou(pred_boxes, target_boxes, x1y1x2y2=False, CIoU=False).detach().clamp(0)
            obj_target_adjusted = target[..., 4][obj_mask] * iou
            obj_pred = pred[..., 4][obj_mask]
            obj_loss = self.bce_obj(obj_pred, obj_target_adjusted) if obj_pred.numel() > 0 else torch.tensor(0.0).to(self.device)
            obj_loss *= self.lambda_obj * self.balance[i]

            noobj_pred = pred[..., 4][noobj_mask & ~ignore_mask]
            noobj_target = target[..., 4][noobj_mask & ~ignore_mask]
            noobj_loss = self.bce_obj(noobj_pred, noobj_target) if noobj_pred.numel() > 0 else torch.tensor(0.0).to(self.device)
            noobj_loss *= self.lambda_noobj

            # Cls Loss
            cls_mask = obj_mask.unsqueeze(-1).expand_as(target[..., 5:])
            cls_pred = pred[..., 5:][cls_mask].reshape(-1, self.num_classes)
            tcls = torch.zeros_like(cls_pred, device=self.device)
            tcls[torch.arange(tcls.size(0)), target[..., 5][obj_mask].long()] = 1.0
            cls_loss = self.bce_cls(cls_pred, tcls) if cls_pred.numel() > 0 else torch.tensor(0.0).to(self.device)
            cls_loss *= self.lambda_cls

            loss = coord_loss + obj_loss + noobj_loss + cls_loss
            total_loss += loss

            coord_loss_total += coord_loss.item() if isinstance(coord_loss, torch.Tensor) else coord_loss
            obj_loss_total += obj_loss.item()
            noobj_loss_total += noobj_loss.item()
            cls_loss_total += cls_loss.item()

        if self.save_logs and batch_idx is not None and epoch is not None:
            if batch_idx % 100 == 0:
                with open(os.path.join("train_eval_metrics", self.filename, "loss.csv"), "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, batch_idx, coord_loss_total, obj_loss_total, noobj_loss_total, cls_loss_total])
                    f.close()

        # print(f"Coord Loss: {coord_loss_total:.2f}, Obj Loss: {obj_loss_total:.2f}, NoObj Loss: {noobj_loss_total:.2f}, Cls Loss: {cls_loss_total:.2f}")
        return total_loss


# if __name__ == "__main__":
#     from models.yolov5m import YOLOv5m

#     model = YOLOv5m(num_classes=1).eval()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)

#     batch_size = 2
#     imgs = torch.randn(batch_size, 3, 640, 640).to(device)
#     targets = [
#         torch.tensor([[0, 0.5, 0.3, 0.04, 0.07]]).to(device),
#         torch.tensor([[0, 0.4, 0.33, 0.037, 0.07]]).to(device)
#     ]

#     with torch.no_grad():
#         preds = model(imgs)

#     loss_fn = YOLOLoss(num_classes=1, anchors=[
#         [1.25, 1.625, 2.0, 3.75, 4.125, 2.875],
#         [1.875, 3.8125, 4.0, 2.0, 3.75, 5.0],
#         [4.0, 7.0, 8.0, 4.0, 6.0, 10.0]
#     ], device=device, save_logs=True, filename="test_run")
#     loss = loss_fn(preds, targets, batch_idx=0, epoch=1)
#     print(f"Total Loss: {loss.item()}")