# utils/augmentations.py
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(img_size=640):
    """
    Trả về transform cho quá trình huấn luyện.
    
    Args:
        img_size (int): Kích thước ảnh đầu vào (mặc định 640)
    
    Returns:
        A.Compose: Chuỗi các phép tăng cường dữ liệu
    """
    return A.Compose([
        # Lật ngang ngẫu nhiên
        A.HorizontalFlip(p=0.5),
        # Thay đổi độ sáng và độ tương phản
        A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        # Thay đổi màu sắc
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
        # Xoay ngẫu nhiên (giữ bounding boxes)
        A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
        # Resize về kích thước cố định
        A.Resize(height=img_size, width=img_size),
        # Chuyển sang tensor
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.5))

def apply_transform(transform, img, boxes):
    """
    Áp dụng transform lên ảnh và bounding boxes.
    
    Args:
        transform (A.Compose): Đối tượng transform từ albumentations
        img (torch.Tensor): Tensor ảnh [C, H, W]
        boxes (torch.Tensor): Tensor bounding boxes [N, 5] với [class_id, x_center, y_center, width, height]
    
    Returns:
        tuple: (img_transformed, boxes_transformed)
    """
    # Chuyển tensor về numpy array cho albumentations
    img = img.numpy().transpose(1, 2, 0)  # [H, W, C]

    if boxes.numel() == 0:
        # Nếu không có bounding box, chỉ transform ảnh
        augmented = transform(image=img)
        img_transformed = torch.from_numpy(augmented['image']).permute(2, 0, 1)
        return img_transformed, boxes

    # Chuẩn bị dữ liệu cho albumentations
    class_labels = boxes[:, 0].long().tolist()  # Lấy class_id
    bboxes = boxes[:, 1:].tolist()  # Lấy [x_center, y_center, width, height]

    # Áp dụng transform
    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)

    # Chuyển kết quả về tensor
    img_transformed = torch.from_numpy(augmented['image']).permute(2, 0, 1)
    boxes_transformed = torch.tensor(augmented['bboxes'], dtype=torch.float32)
    if boxes_transformed.numel() > 0:
        boxes_transformed = torch.cat([torch.tensor(augmented['class_labels'], dtype=torch.float32).unsqueeze(1), boxes_transformed], dim=1)

    return img_transformed, boxes_transformed

# Test augmentations
if __name__ == "__main__":
    import cv2

    # Tạo transform
    transform = get_train_transform(img_size=640)

    # Ảnh mẫu và bounding boxes mẫu (dựa trên dữ liệu của bạn)
    img = np.random.rand(640, 640, 3) * 255  # Ảnh giả lập
    boxes = torch.tensor([
        [0, 0.50146484375, 0.2964860907759883, 0.0361328125, 0.0746705710102489],
        [0, 0.4150390625, 0.3323572474377745, 0.037109375, 0.07027818448023426]
    ])

    # Áp dụng transform
    img_transformed, boxes_transformed = apply_transform(transform, torch.from_numpy(img).permute(2, 0, 1), boxes)

    print(f"Original image shape: {img.shape}")
    print(f"Transformed image shape: {img_transformed.shape}")
    print(f"Original boxes shape: {boxes.shape}")
    print(f"Transformed boxes shape: {boxes_transformed.shape if boxes_transformed.numel() > 0 else 'Empty'}")

    # Hiển thị ảnh (tùy chọn)
    img_cv = img_transformed.permute(1, 2, 0).numpy() * 255
    cv2.imwrite("augmented_image.jpg", img_cv)