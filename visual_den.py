import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from model.MMFFNet import MMFFNet


INPUT_H = 480
INPUT_W = 640


def load_image_as_tensor(img_path, size=(480, 640)):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),   # (H, W)
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img


def normalize_map(x):
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def load_gt_from_points(gt_path, orig_img_size, out_size=(640, 480), sigma=8):
    pts = np.load(gt_path).astype(np.float32)
    gt_count = len(pts)

    orig_w, orig_h = orig_img_size
    out_w, out_h = out_size

    point_map = np.zeros((out_h, out_w), dtype=np.float32)

    for p in pts:
        x, y = p[0], p[1]
        x = x * out_w / orig_w
        y = y * out_h / orig_h

        x = int(round(x))
        y = int(round(y))

        x = max(0, min(x, out_w - 1))
        y = max(0, min(y, out_h - 1))

        point_map[y, x] = 1.0

    gt_map = cv2.GaussianBlur(point_map, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gt_map = normalize_map(gt_map)

    return gt_map, gt_count


def save_four_panel(rgb_tensor, t_tensor, pred_tensor, gt_map, gt_count, save_path):
    rgb = rgb_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    rgb = normalize_map(rgb)

    t = t_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    t = normalize_map(t)

    pred_count = pred_tensor[0, 0].sum().item()

    pred_map = F.interpolate(
        pred_tensor,
        size=(INPUT_H, INPUT_W),   # (H, W)
        mode='bilinear',
        align_corners=False
    )
    pred_map = pred_map[0, 0].detach().cpu().numpy()
    pred_map = normalize_map(pred_map)
    pred_map = cv2.GaussianBlur(pred_map, (9, 9), 0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb)
    axes[0].axis('off')
    axes[0].set_title('RGB')

    axes[1].imshow(t)
    axes[1].axis('off')
    axes[1].set_title('T')

    axes[2].imshow(gt_map, cmap='jet')
    axes[2].axis('off')
    axes[2].set_title(f'GT Density\nCount: {gt_count}')

    axes[3].imshow(pred_map, cmap='jet')
    axes[3].axis('off')
    axes[3].set_title(f'Pred Density\nCount: {pred_count:.1f}')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()


if __name__ == "__main__":
    os.makedirs("vis_results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = MMFFNet().to(device)
    checkpoint = torch.load("./trained_model/RGBTCC/300_67_5.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    rgb_path = "./dataset/RGBT_CC/test/1197_RGB.jpg"
    t_path   = "./dataset/RGBT_CC/test/1197_T.jpg"
    gt_path  = "./dataset/RGBT_CC/test/1197_GT.npy"

    orig_img = Image.open(rgb_path)
    orig_w, orig_h = orig_img.size

    rgb = load_image_as_tensor(rgb_path, size=(INPUT_H, INPUT_W)).to(device)
    t   = load_image_as_tensor(t_path, size=(INPUT_H, INPUT_W)).to(device)

    with torch.no_grad():
        pred, pred_norm = model(rgb, t)

    print("pred shape:", pred.shape)
    print("pred count:", pred[0, 0].sum().item())

    gt_map, gt_count = load_gt_from_points(
        gt_path,
        orig_img_size=(orig_w, orig_h),
        out_size=(INPUT_W, INPUT_H),   # (W, H)
        sigma=8
    )

    print("gt count:", gt_count)

    save_four_panel(
        rgb_tensor=rgb,
        t_tensor=t,
        pred_tensor=pred,
        gt_map=gt_map,
        gt_count=gt_count,
        save_path="vis_results/rgb_t_gt_pred.png"
    )

    print("结果已保存到 vis_results/rgb_t_gt_pred.png")