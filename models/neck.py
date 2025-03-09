import torch
import torch.nn as nn

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 9, 13]):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.silu = nn.SiLU(inplace=True)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])
        self.conv2 = nn.Conv2d(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.silu(self.bn1(self.conv1(x)))
        pool_outputs = [x]
        for pool in self.pools:
            pool_outputs.append(pool(x))
        x = torch.cat(pool_outputs, 1)
        x = self.silu(self.bn2(self.conv2(x)))
        return x

def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True)
    )

class PANet(nn.Module):
    def __init__(self, channels_list=[256, 512, 1024], width=1.0):
        super().__init__()
        channels_list = [int(c * width) for c in channels_list]  # [256, 512, 1024] với width=1.0

        # SPPF cho P5
        self.sppf = SPPF(channels_list[2], channels_list[2])  # 1024 -> 1024

        # FPN: Top-down path
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_p5_to_p4 = conv_block(channels_list[2], channels_list[1], 1)  # 1024 -> 512
        self.conv_p4_fpn = conv_block(channels_list[1] + channels_list[1], channels_list[1], 3, padding=1)  # 512 + 512 -> 512
        self.conv_p4_to_p3 = conv_block(channels_list[1], channels_list[0], 1)  # 512 -> 256
        self.conv_p3_fpn = conv_block(channels_list[0] + channels_list[0], channels_list[0], 3, padding=1)  # 256 + 256 -> 256

        # PAN: Bottom-up path
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_p3_to_p4 = conv_block(channels_list[0], channels_list[1], 3, padding=1)  # 256 -> 512
        self.conv_p4_pan = conv_block(channels_list[1] + channels_list[1], channels_list[1], 3, padding=1)  # 512 + 512 -> 512
        self.conv_p4_to_p5 = conv_block(channels_list[1], channels_list[2], 3, padding=1)  # 512 -> 1024
        self.conv_p5_pan = conv_block(channels_list[2] + channels_list[2], channels_list[2], 3, padding=1)  # 1024 + 1024 -> 1024

    def forward(self, p3, p4, p5):
        # SPPF cho P5
        p5 = self.sppf(p5)  # 1024 -> 1024

        # FPN: Top-down
        p5_up = self.conv_p5_to_p4(p5)  # 1024 -> 512
        p5_up = self.up(p5_up)
        p4 = torch.cat([p4, p5_up], 1)  # 512 + 512 = 1024
        p4 = self.conv_p4_fpn(p4)       # 1024 -> 512

        p4_up = self.conv_p4_to_p3(p4)  # 512 -> 256
        p4_up = self.up(p4_up)
        p3 = torch.cat([p3, p4_up], 1)  # 256 + 256 = 512
        p3 = self.conv_p3_fpn(p3)       # 512 -> 256

        # PAN: Bottom-up
        p3_down = self.conv_p3_to_p4(p3)  # 256 -> 512
        p3_down = self.down(p3_down)
        p4 = torch.cat([p4, p3_down], 1)  # 512 + 512 = 1024
        p4 = self.conv_p4_pan(p4)         # 1024 -> 512

        p4_down = self.conv_p4_to_p5(p4)  # 512 -> 1024
        p4_down = self.down(p4_down)
        p5 = torch.cat([p5, p4_down], 1)  # 1024 + 1024 = 2048
        p5 = self.conv_p5_pan(p5)         # 2048 -> 1024

        return p3, p4, p5  # [256, 512, 1024]

# if __name__ == "__main__":
#     panet = PANet(channels_list=[256, 512, 1024], width=1.0)
#     p3 = torch.randn(1, 256, 80, 80)
#     p4 = torch.randn(1, 512, 40, 40)
#     p5 = torch.randn(1, 1024, 20, 20)
#     p3_out, p4_out, p5_out = panet(p3, p4, p5)
#     print(p3_out.shape)  # [1, 256, 80, 80]
#     print(p4_out.shape)  # [1, 512, 40, 40]
#     print(p5_out.shape)  # [1, 1024, 20, 20]