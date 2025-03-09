import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024], num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Tạo head cho từng scale (P3, P4, P5)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.SiLU(),
                nn.Conv2d(in_channels // 2, num_anchors * (5 + num_classes), 1)
            ) for in_channels in in_channels_list
        ])

    def forward(self, inputs):
        # inputs: list of tensors [P3, P4, P5]
        outputs = []
        for i, x in enumerate(inputs):
            out = self.heads[i](x)  # Áp dụng head cho từng scale
            batch, _, height, width = out.shape
            out = out.view(batch, self.num_anchors, 5 + self.num_classes, height, width)
            out = out.permute(0, 1, 3, 4, 2)  # [batch, num_anchors, height, width, 5 + num_classes]
            outputs.append(out)
        return outputs  # [P3_preds, P4_preds, P5_preds]


# if __name__ == "__main__":
#     head = YOLOHead(in_channels_list=[256, 512, 1024], num_classes=80, num_anchors=3)
#     p3 = torch.randn(1, 256, 80, 80)  # Từ PANet
#     p4 = torch.randn(1, 512, 40, 40)
#     p5 = torch.randn(1, 1024, 20, 20)
#     preds = head([p3, p4, p5])
#     for i, pred in enumerate(preds):
#         print(f"Scale {i+3}: {pred.shape}")  # P3, P4, P5