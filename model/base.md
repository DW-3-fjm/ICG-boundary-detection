
# U-Net ICG Dummy Implementation

## Context

### Indocyanine Green Memory Technology
- Clinical Pain Point: Currently, indocyanine green (ICG) imaging is used during surgery, but the duration is short, making it difficult to record important resection margins.
- Technical Goals and Route: Incorporate video of lung segmentectomy with ICG imaging, label the fluorescence margins before and after imaging, construct a model to predict margins in real-time, and overlay the boundary lines on the laparoscopic image.

> Input
ICG boundary after imaging
> Output
Display boundary in the absence of ICG imaging
Indocyanine Green (ICG) is a fluorescent dye used in medical imaging to visualize blood flow and tissue perfusion. 'Lung segmentectomy' is a surgical procedure to remove a segment of the lung.

```python
# Data preprocessing:

##### HERE GOES THE PREPROCESSING

# U-Net main algorithm:
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder (Expanding Path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)  # 512 (from up) + 512 (from skip) = 1024
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)   # 256 + 256 = 512
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)   # 128 + 128 = 256
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)     # 64 + 64 = 128
        
        # Final output projection
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder & Save Skip Connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder & Concatenate Skips
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits

# If green detected, we have the real-time contour
# If ICG disappeared, we have the memory contour

# Movement prediction algorithm
# Kalman filter or Optical flow
import cv2
import numpy as np

class ContourKalmanTracker:
    def __init__(
        self, initial_contour, process_noise=1e-2, measurement_noise=1e-1
    ):
        self.num_points = len(initial_contour)
        self.contour = initial_contour.reshape(-1, 2).astype(np.float32)

        self.kalman_filters = []
        for i in range(self.num_points):
            kf = cv2.KalmanFilter(4, 2)

            # State Transition Matrix (Constant Velocity Model):
            # x_k = x_{k-1} + vx * dt
            kf.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], np.float32)
```