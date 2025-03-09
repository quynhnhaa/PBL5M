# # train.py
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from models.yolov5m import YOLOv5m
# from utils.dataset import YOLODataset, collate_fn
# from utils.loss import YOLOLoss
# from config import Config
# import os
# # Tạo DataLoader cho tập validation
# val_dataset = YOLODataset(
#     img_dir=Config.VAL_IMG_DIR,
#     label_dir=Config.VAL_LABEL_DIR,
#     anchors=Config.ANCHORS,
#     img_size=Config.IMG_SIZE
# )
# val_loader = DataLoader(
#     dataset=val_dataset,
#     batch_size=Config.BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn
# )

# def validate(val_loader, model, loss_fn):
#     """
#     Hàm tính loss trung bình trên tập validation.
    
#     Args:
#         val_loader (DataLoader): DataLoader cho tập validation.
#         model (nn.Module): Mô hình YOLOv5m.
#         loss_fn (YOLOLoss): Hàm loss.
    
#     Returns:
#         float: Loss trung bình trên tập validation.
#     """
#     model.eval()  # Chuyển mô hình sang chế độ đánh giá
#     total_val_loss = 0.0
#     num_batches = 0

#     with torch.no_grad():  # Tắt gradient để tiết kiệm bộ nhớ
#         for images, targets in val_loader:
#             images = images.to(Config.DEVICE)
#             targets = [target.to(Config.DEVICE) for target in targets]

#             # Forward pass
#             preds = model(images)  # Dự đoán từ mô hình
#             val_loss = loss_fn(preds, targets, img_size=Config.IMG_SIZE)  # Tính loss

#             total_val_loss += val_loss.item()
#             num_batches += 1

#     avg_val_loss = total_val_loss / num_batches
#     print(f"Validation Loss: {avg_val_loss:.4f}")
#     return avg_val_loss

# def train():
#     model = YOLOv5m(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

#     train_dataset = YOLODataset(
#         img_dir=Config.TRAIN_IMG_DIR,
#         label_dir=Config.TRAIN_LABEL_DIR,
#         anchors=Config.ANCHORS,
#         img_size=Config.IMG_SIZE
#     )

#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=True,
#         collate_fn=collate_fn
#     )

#     loss_fn = YOLOLoss(
#         num_classes=Config.NUM_CLASSES,
#         anchors=Config.ANCHORS,
#         strides=Config.STRIDES,
#         device=Config.DEVICE,
#         save_logs=True,
#         filename="widerface_run"
#     )

#     def train_epoch(epoch):
#         model.train()
#         total_loss = 0.0

#         for batch_idx, (images, targets) in enumerate(train_loader):
#             images = images.to(Config.DEVICE)
#             targets = [target.to(Config.DEVICE) for target in targets]

#             preds = model(images)
#             loss = loss_fn(preds, targets, img_size=Config.IMG_SIZE, batch_idx=batch_idx, epoch=epoch)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             if batch_idx % 10 == 0:
#                 print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Average Loss: {avg_loss:.4f}")
#         return avg_loss

#     best_loss = float("inf")
#     for epoch in range(Config.EPOCHS):
#         avg_loss = train_epoch(epoch)
#         # Thêm validation sau mỗi epoch
#         avg_val_loss = validate(val_loader, model, loss_fn)

#         if avg_val_loss < best_loss:
#             best_loss = avg_val_loss
#             torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
#             print(f"Saved best model with validation loss: {best_loss:.4f}")

#         model_save_path = Config.MODEL_SAVE_PATH.format(epoch=epoch + 1)
#         torch.save(model.state_dict(), model_save_path)
#         print(f"Saved model to {model_save_path}")

# if __name__ == "__main__":
#     train()
# train.py
# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.yolov5m import YOLOv5m
from utils.dataset import YOLODataset, collate_fn
from utils.loss import YOLOLoss
from utils.postprocess import cells_to_bboxes
from utils.evaluate import calculate_map
from config import Config
import os
import numpy as np
import csv

# Tạo DataLoader cho tập validation
val_dataset = YOLODataset(
    img_dir=Config.VAL_IMG_DIR,
    label_dir=Config.VAL_LABEL_DIR,
    anchors=Config.ANCHORS,
    img_size=Config.IMG_SIZE
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

def validate(val_loader, model, loss_fn):
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    all_pred_bboxes = []
    all_true_bboxes = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)
            targets = [target.to(Config.DEVICE) for target in targets]

            preds = model(images)
            val_loss = loss_fn(preds, targets, img_size=Config.IMG_SIZE)

            pred_bboxes = cells_to_bboxes(
                preds,
                anchors=Config.ANCHORS,
                strides=Config.STRIDES,
                img_size=Config.IMG_SIZE,
                conf_thres=0.3,
                iou_thres=0.5
            )
            all_pred_bboxes.extend(pred_bboxes)
            all_true_bboxes.extend(targets)

            total_val_loss += val_loss.item()
            num_batches += 1

    avg_val_loss = total_val_loss / num_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")

    mAP = calculate_map(
        pred_bboxes=all_pred_bboxes,
        true_bboxes=all_true_bboxes,
        iou_thres=0.5,
        num_classes=Config.NUM_CLASSES
    )

    return avg_val_loss, mAP

def train():
    model = YOLOv5m(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    train_dataset = YOLODataset(
        img_dir=Config.TRAIN_IMG_DIR,
        label_dir=Config.TRAIN_LABEL_DIR,
        anchors=Config.ANCHORS,
        img_size=Config.IMG_SIZE
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    loss_fn = YOLOLoss(
        num_classes=Config.NUM_CLASSES,
        anchors=Config.ANCHORS,
        strides=Config.STRIDES,
        device=Config.DEVICE,
        save_logs=True,
        filename="widerface_run"
    )

    # Tạo hoặc mở file CSV để ghi trọng số
    weights_csv_path = os.path.join("train_eval_metrics", "widerface_run", "weights.csv")
    os.makedirs(os.path.dirname(weights_csv_path), exist_ok=True)
    with open(weights_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch",
            "Stem_Mean", "Stem_Std",
            "Stage3_Mean", "Stage3_Std",
            "Stage5_Mean", "Stage5_Std",
            "Neck_P5_to_P4_Mean", "Neck_P5_to_P4_Std",
            "Head_P3_Mean", "Head_P3_Std"
        ])

    def train_epoch(epoch):
        model.train()
        total_loss = 0.0
        weights_stats = {
            "stem_mean": [], "stem_std": [],
            "stage3_mean": [], "stage3_std": [],
            "stage5_mean": [], "stage5_std": [],
            "neck_p5_to_p4_mean": [], "neck_p5_to_p4_std": [],
            "head_p3_mean": [], "head_p3_std": []
        }

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(Config.DEVICE)
            targets = [target.to(Config.DEVICE) for target in targets]

            preds = model(images)
            loss = loss_fn(preds, targets, img_size=Config.IMG_SIZE, batch_idx=batch_idx, epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Lấy trọng số từ các lớp
            # 1. Backone - Stem
            stem_weight = model.backbone.stem[0].weight.data.cpu().numpy()
            weights_stats["stem_mean"].append(np.mean(np.abs(stem_weight)))
            weights_stats["stem_std"].append(np.std(stem_weight))

            # 2. Backbone - Stage3
            stage3_weight = model.backbone.stage3[0].weight.data.cpu().numpy()
            weights_stats["stage3_mean"].append(np.mean(np.abs(stage3_weight)))
            weights_stats["stage3_std"].append(np.std(stage3_weight))

            # 3. Backbone - Stage5
            stage5_weight = model.backbone.stage5[0].weight.data.cpu().numpy()
            weights_stats["stage5_mean"].append(np.mean(np.abs(stage5_weight)))
            weights_stats["stage5_std"].append(np.std(stage5_weight))

            # 4. Neck - conv_p5_to_p4
            neck_p5_to_p4_weight = model.neck.conv_p5_to_p4[0].weight.data.cpu().numpy()
            weights_stats["neck_p5_to_p4_mean"].append(np.mean(np.abs(neck_p5_to_p4_weight)))
            weights_stats["neck_p5_to_p4_std"].append(np.std(neck_p5_to_p4_weight))

            # 5. Head - heads[0]
            head_p3_weight = model.head.heads[0][0].weight.data.cpu().numpy()
            weights_stats["head_p3_mean"].append(np.mean(np.abs(head_p3_weight)))
            weights_stats["head_p3_std"].append(np.std(head_p3_weight))

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Average Loss: {avg_loss:.4f}")

        # Tính trung bình qua các batch
        return (avg_loss,
                np.mean(weights_stats["stem_mean"]), np.mean(weights_stats["stem_std"]),
                np.mean(weights_stats["stage3_mean"]), np.mean(weights_stats["stage3_std"]),
                np.mean(weights_stats["stage5_mean"]), np.mean(weights_stats["stage5_std"]),
                np.mean(weights_stats["neck_p5_to_p4_mean"]), np.mean(weights_stats["neck_p5_to_p4_std"]),
                np.mean(weights_stats["head_p3_mean"]), np.mean(weights_stats["head_p3_std"]))

    best_val_loss = float("inf")
    best_mAP = 0.0
    patience = 5
    trigger_times = 0

    for epoch in range(Config.EPOCHS):
        avg_train_loss, stem_mean, stem_std, stage3_mean, stage3_std, stage5_mean, stage5_std, neck_p5_to_p4_mean, neck_p5_to_p4_std, head_p3_mean, head_p3_std = train_epoch(epoch)

        # Ghi trọng số vào CSV
        with open(weights_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                stem_mean, stem_std,
                stage3_mean, stage3_std,
                stage5_mean, stage5_std,
                neck_p5_to_p4_mean, neck_p5_to_p4_std,
                head_p3_mean, head_p3_std
            ])

        avg_val_loss, mAP = validate(val_loader, model, loss_fn)

        scheduler.step(avg_val_loss)

        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"No improvement in validation loss, trigger times: {trigger_times}")

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model_map.pth"))
            print(f"Saved best model with mAP: {best_mAP:.4f}")

        if trigger_times >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            break

        model_save_path = Config.MODEL_SAVE_PATH.format(epoch=epoch + 1)
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path}")

if __name__ == "__main__":
    train()