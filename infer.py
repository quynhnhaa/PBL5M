# infer.py
import torch
import cv2
import numpy as np
import os
from models.yolov5m import YOLOv5m
from utils.postprocess import cells_to_bboxes
from config import Config

def predict(image_path, model, conf_thres=0.3, iou_thres=0.5):
    """
    Dự đoán bounding box trên ảnh.

    Args:
        image_path (str): Đường dẫn đến ảnh.
        model (nn.Module): Mô hình đã huấn luyện.
        conf_thres (float): Ngưỡng confidence để lọc box.
        iou_thres (float): Ngưỡng IoU cho NMS.

    Returns:
        list: Danh sách bounding box [x1, y1, x2, y2, score, class].
    """
    model.eval()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh tại {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (Config.IMG_SIZE, Config.IMG_SIZE))

    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        preds = model(image_tensor)
        pred_bboxes = cells_to_bboxes(
            preds,
            anchors=Config.ANCHORS,
            strides=Config.STRIDES,
            img_size=Config.IMG_SIZE,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )[0]

    return pred_bboxes

def visualize(image_path, pred_bboxes, results_dir="WIDER_test/results"):
    """
    Vẽ bounding box lên ảnh và lưu vào thư mục kết quả.

    Args:
        image_path (str): Đường dẫn đến ảnh gốc.
        pred_bboxes (list): Danh sách bounding box.
        results_dir (str): Thư mục lưu kết quả.
    """
    image = cv2.imread(image_path)

    # Vẽ bounding box lên ảnh
    for box in pred_bboxes:
        x1, y1, x2, y2, score, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tạo đường dẫn lưu kết quả
    relative_path = os.path.relpath(image_path, "WIDER_test/images")  # Lấy đường dẫn tương đối (ví dụ: 0--Parade/0_Parade_marchingband_1_9.jpg)
    output_path = os.path.join(results_dir, relative_path)
    output_dir = os.path.dirname(output_path)

    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lưu ảnh kết quả
    output_path = os.path.splitext(output_path)[0] + "_result.jpg"
    cv2.imwrite(output_path, image)
    print(f"Ảnh kết quả đã được lưu tại: {output_path}")

if __name__ == "__main__":
    # Load mô hình đã huấn luyện
    model = YOLOv5m(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Thư mục chứa ảnh test
    test_images_dir = "WIDER_test/images/"

    # Duyệt qua các thư mục con trong WIDER_test/images/
    for category in os.listdir(test_images_dir):
        category_path = os.path.join(test_images_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Duyệt qua các ảnh trong thư mục con
        for image_name in os.listdir(category_path):
            if not image_name.endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(category_path, image_name)
            print(f"\nDự đoán trên ảnh: {image_path}")

            # Dự đoán
            try:
                pred_bboxes = predict(image_path, model, conf_thres=0.2, iou_thres=0.5)
                print(f"Detected boxes: {len(pred_bboxes)}")
                for box in pred_bboxes:
                    print(f"Box: x1={box[0]:.2f}, y1={box[1]:.2f}, x2={box[2]:.2f}, y2={box[3]:.2f}, score={box[4]:.2f}, class={box[5]}")

                # Visualize nếu có box
                if len(pred_bboxes) > 0:
                    visualize(image_path, pred_bboxes, results_dir="WIDER_test/results")
                else:
                    print("Không có bounding box nào được phát hiện.")
            except Exception as e:
                print(f"Lỗi khi dự đoán: {e}")