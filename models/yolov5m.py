import torch
import torch.nn as nn

from models.backbone import CSPDarknet53  # Phiên bản trả về [p3, p4, p5]
from models.neck import PANet            # Phiên bản nhận channels_list và out_channels
from models.head import YOLOHead         # Phiên bản xử lý danh sách [p3, p4, p5]

class YOLOv5m(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = CSPDarknet53(in_channels=3, depth=1.0, width=1.0)
        self.neck = PANet(channels_list=[256, 512, 1024], width=1.0)
        self.head = YOLOHead(in_channels_list=[256, 512, 1024], num_classes=num_classes, num_anchors=3)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        preds = self.head([p3, p4, p5])
        return preds

# # Kiểm tra
# model = YOLOv5m(num_classes=80)
# x = torch.randn(1, 3, 640, 640)
# preds = model(x)
# for i, pred in enumerate(preds):
#     print(f"P{i+3}: {pred.shape}")