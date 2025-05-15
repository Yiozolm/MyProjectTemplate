import os
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def load_images(folder, transform, max_images=None):
    images = []
    filenames = sorted(os.listdir(folder))
    if max_images:
        filenames = filenames[:max_images]
    for fname in tqdm(filenames, desc=f"Loading {folder}"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, fname)).convert('RGB')
            images.append(transform(img))
    return torch.stack(images), filenames

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径
folder_real = '/hy-tmp/Kodak24/'
folder_fake = './Eval/bshift1-epoch3/1'

# transforms
transform_fid = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

transform_metric = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 加载图像
real_fid, _ = load_images(folder_real, transform_fid)
fake_fid, _ = load_images(folder_fake, transform_fid)

real_metric, _ = load_images(folder_real, transform_metric)
fake_metric, _ = load_images(folder_fake, transform_metric)

# FID
fid = FrechetInceptionDistance(normalize=True).to(device)
fid.update(real_fid.to(device), real=True)
fid.update(fake_fid.to(device), real=False)
fid_score = fid.compute().item()

# LPIPS（支持 alex / vgg / squeeze）
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
lpips_scores = lpips(real_metric.to(device), fake_metric.to(device))

# DISTS
dists = DeepImageStructureAndTextureSimilarity().to(device)
dists_scores = dists(fake_metric.to(device), real_metric.to(device))

# 输出
print(f"\n📊 Evaluation Results (TorchMetrics):")
print(f"FID   Score : {fid_score:.4f}")
print(f"LPIPS Score : {lpips_scores:.4f}")
print(f"DISTS Score : {dists_scores:.4f}")