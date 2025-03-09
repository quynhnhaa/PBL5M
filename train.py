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

            # Chuyển dự đoán thành bounding box để tính mAP
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

    # Tính mAP
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

    # Thêm scheduler mà không dùng verbose
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

    def train_epoch(epoch):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(Config.DEVICE)
            targets = [target.to(Config.DEVICE) for target in targets]

            preds = model(images)
            loss = loss_fn(preds, targets, img_size=Config.IMG_SIZE, batch_idx=batch_idx, epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Average Loss: {avg_loss:.4f}")
        return avg_loss

    best_val_loss = float("inf")
    best_mAP = 0.0
    patience = 5  # Số epoch chờ trước khi dừng
    trigger_times = 0

    for epoch in range(Config.EPOCHS):
        avg_train_loss = train_epoch(epoch)

        # Chạy validation và tính mAP
        avg_val_loss, mAP = validate(val_loader, model, loss_fn)

        # Cập nhật scheduler
        scheduler.step(avg_val_loss)

        # In learning rate hiện tại
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Lưu mô hình nếu validation loss cải thiện
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"No improvement in validation loss, trigger times: {trigger_times}")

        # Lưu mô hình nếu mAP cải thiện
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model_map.pth"))
            print(f"Saved best model with mAP: {best_mAP:.4f}")

        # Early stopping
        if trigger_times >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            break

        # Lưu mô hình sau mỗi epoch
        model_save_path = Config.MODEL_SAVE_PATH.format(epoch=epoch + 1)
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path}")

if __name__ == "__main__":
    train()