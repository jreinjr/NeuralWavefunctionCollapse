import numpy as np
import cv2
import torchvision.io as io

# 1. Produce and randomize the array
frames = [np.random.rand(256, 256) for _ in range(60)]
frames = [((frame * 255).astype(np.uint8)) for frame in frames]  # Convert to 8-bit

# 2. Save as a video using OpenCV
fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # FFV1 is a lossless codec
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (256, 256), False)  # False for grayscale

for frame in frames:
    out.write(frame)

out.release()

# 3. Load the video using torchvision
video_tensor, audio_tensor, info = io.read_video('output.avi', pts_unit='sec')

# Convert to desired shape: [T, H, W] (from T, H, W, C)
video_tensor = video_tensor.squeeze(-1)  # Removes channel dimension since it's 1 for grayscale

print(video_tensor.shape)  # Expected: torch.Size([60, 256, 256])
