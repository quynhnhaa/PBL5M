import torch
import torch.nn as nn

# Khối C3 (Cross Stage Partial) 
class C3(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.silu = nn.SiLU()
        self.bottlenecks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            ) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(hidden_channels * 2, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.silu(self.bn1(self.conv1(x)))  # Nhánh 1
        x2 = self.silu(self.bn2(self.conv2(x)))  # Nhánh 2
        x2 = self.bottlenecks(x2)                # Xử lý qua Bottleneck
        x = torch.cat([x1, x2], dim=1)           # Nối hai nhánh
        x = self.silu(self.bn_out(self.conv_out(x)))  # Tổng hợp
        return x

# CSPDarknet53 
class CSPDarknet53(nn.Module):
    def __init__(self, in_channels=3, depth=1.0, width=1.0):
        super().__init__()
        base_channels = int(64 * width)  # Số kênh cơ bản
        base_depth = max(round(3 * depth), 1)  # Độ sâu CSP

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 6, 2, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU()
        )

        # Các stage trả về P3, P4, P5
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU()
        )
        self.stage2 = C3(base_channels * 2, base_channels * 2, base_depth)  # P2
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(),
            C3(base_channels * 4, base_channels * 4, base_depth * 3)  # P3
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.SiLU(),
            C3(base_channels * 8, base_channels * 8, base_depth * 3)  # P4
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 16),
            nn.SiLU(),
            C3(base_channels * 16, base_channels * 16, base_depth)  # P5
        )

    def forward(self, x):
        x = self.stem(x)         # 640x640 -> 320x320
        x = self.stage1(x)       # 320x320 -> 160x160
        p2 = self.stage2(x)      # 160x160
        p3 = self.stage3(p2)     # 80x80
        p4 = self.stage4(p3)     # 40x40
        p5 = self.stage5(p4)     # 20x20
        return p3, p4, p5        # Trả về P3, P4, P5


# if __name__ == "__main__":
#     model = CSPDarknet53(depth=1.0, width=1)
#     x = torch.randn(1, 3, 640, 640)
#     p3, p4, p5 = model(x)
#     print(p3.shape)  # [1, 256, 80, 80]
#     print(p4.shape)  # [1, 512, 40, 40]
#     print(p5.shape)  # [1, 1024, 20, 20]