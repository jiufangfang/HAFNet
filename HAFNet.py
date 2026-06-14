import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from model.FFM import FFM
from model.PFD import PFD
from timm import create_model
import numpy as np
from torch.utils.data import DataLoader

# 生成预测并保存
def inference_and_save(model, test_loader, device='cuda', save_path='predictions.npy'):

    model = model.to(device)
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for batch_idx, (rgb_images, t_images, _) in enumerate(test_loader):
            # 数据移到GPU
            rgb_images = rgb_images.to(device)
            t_images = t_images.to(device)

            # 前向传播（你的模型返回两个值）
            density_map, density_map_normed = model(rgb_images, t_images)

            # 计算预测人数（对密度图求和）
            batch_predictions = density_map.sum(dim=(1, 2, 3)).cpu().numpy()
            all_predictions.extend(batch_predictions)

            if batch_idx % 50 == 0:
                print(f"已处理 {batch_idx}/{len(test_loader)} 个batch")

    # 保存为npy文件
    predictions_array = np.array(all_predictions)
    np.save(save_path, predictions_array)  # 最终保存位置
    print(f"已保存 {len(predictions_array)} 个预测结果到 {save_path}")

    return predictions_array

model = HAFNet()
model.load_state_dict(torch.load('best_model.pth'))
test_loader = YourDataLoader()  # 需返回 (rgb_img, t_img, gt_count)

predictions = inference_and_save(model, test_loader, device='cuda')

class HAFNet(nn.Module):
    def __init__(self):
        super(HAFNet, self).__init__()
        swin = timm.create_model(
            'swin_small_patch4_window7_224',
            pretrained=True,
            img_size=(480, 640)
        )
        self.swin_features = nn.Sequential(*list(swin.children())[:-1])

        self.ffm = FFM(channels=384)  # 768
        self.fdm = PFD(384, 192, 32)  # 768 192 32

    def forward(self, rgb, t):
        # Swin 默认可以接受任意 H, W（patch size 可整除）
        rgb_feat = self.swin_features(rgb)
        t_feat = self.swin_features(t)
        # 转换为 [B, C, H, W]
        rgb_feat = rgb_feat.permute(0, 3, 1, 2).contiguous()
        t_feat = t_feat.permute(0, 3, 1, 2).contiguous()

        share = self.ffm(rgb_feat, t_feat)
        x = self.fdm(share)
        B, C, H, W = x.size()
        x_sum = x.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_normed = x / (x_sum + 1e-6)

        return x, x_normed




if __name__ == '__main__':
    from thop import profile


    model = HAFNet().cuda()
    # print(model)
    model.eval()

    x1 = torch.randn((1, 3, 480, 640)).cuda()
    x2 = torch.randn((1, 3, 480, 640)).cuda()
    out = model(x1, x2)
    print(out[0].shape, out[1].shape)
    flops, params = profile(model, inputs=(x1, x2))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # x1 = torch.rand(2, 3, 256, 256).cuda()
    # x2 = torch.rand(2, 3, 256, 256).cuda()
    # y1, y2 = model(x1, x2)
    # print(y1.shape, y2.shape)