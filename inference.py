import os
import torch
import cv2
import numpy as np

from models.enet import ENet

# -------------------------
# 1. Load Model
# -------------------------
weights = torch.load("./save/ENet_CamVid/ENet", map_location=torch.device("cpu"))
state_dict = weights["state_dict"]

num_classes = 12
model = ENet(num_classes=num_classes)
model.load_state_dict(state_dict)
model.eval()

print("✅ Model loaded. Ready for batch inference.")

# -------------------------
# 2. Input / Output folders
# -------------------------
input_dir = "./v220-test"
output_dir = "./v220-test-predict"
os.makedirs(output_dir, exist_ok=True)

# Overlay transparency
ALPHA = 0.5  # 0=only original image, 1=only mask


# -------------------------
# 3. Process all BMP images
# -------------------------
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".bmp"):
        continue

    name_no_ext = os.path.splitext(filename)[0]
    input_path = os.path.join(input_dir, filename)

    # Output files
    output_mask_path = os.path.join(output_dir, f"predict_{name_no_ext}.png")
    output_overlay_path = os.path.join(output_dir, f"overlay_{name_no_ext}.png")

    print("➡️ Processing:", input_path)

    img = cv2.imread(input_path)
    if img is None:
        print("⚠️ Failed to read:", input_path)
        continue

    h, w, _ = img.shape

    # Resize → ENet input
    img_resized = cv2.resize(img, (480, 360))
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(2, 0, 1)

        # 保存预处理前的tensor
    np.save(os.path.join(output_dir, f"preprocessed_{name_no_ext}.npy"), img_tensor)

    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)

    # -------------------------
    # Inference
    # -------------------------
    with torch.no_grad():
        pred = model(img_tensor)
        # 保存模型原始输出（logits）
        np.save(os.path.join(output_dir, f"model_output_{name_no_ext}.npy"), pred.numpy())
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()


    # 保存argmax后的预测结果
    np.save(os.path.join(output_dir, f"argmax_output_{name_no_ext}.npy"), pred)
    # -------------------------
    # Create mask (12 → 4 classes)
    # -------------------------
    mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    mask[pred == 0] = [255, 0, 0]     # Blue: sky
    mask[pred == 5] = [0, 255, 0]     # Green: vegetation
    mask[pred == 9] = [0, 0, 255]     # Red: person

    # Background
    bg = (pred != 0) & (pred != 5) & (pred != 9)
    mask[bg] = [0, 0, 0]

    # Resize mask back to original image size
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # -------------------------
    # Save segmentation mask
    # -------------------------
    cv2.imwrite(output_mask_path, mask_resized)
    print("✅ Mask saved:", output_mask_path)

    # -------------------------
    # 4. Create overlay (blend original + mask)
    # -------------------------
    overlay = cv2.addWeighted(img, 1 - ALPHA, mask_resized, ALPHA, 0)

    # Save overlay image
    cv2.imwrite(output_overlay_path, overlay)
    print("✅ Overlay saved:", output_overlay_path)


print("\n🎉 All images processed with overlay output!")
