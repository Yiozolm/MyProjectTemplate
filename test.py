import math
import os
import time
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
import lpips
from DISTS_pytorch import DISTS
from pytorch_fid import fid_score

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    I = np.array(img).astype(np.float32)
    output = torch.as_tensor(I).permute(2, 0, 1)
    return output


def compute_metrics(
        org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = org.round()
    rec = rec.round()
    metrics["psnr-rgb"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    metrics["log_ssim"] = -10 * np.log10(1 - metrics["ms-ssim"])
    return metrics


def eval_dist(origin_path, gene_path):
    device = torch.device('cuda')
    D = DISTS(load_weights=True).to(device)
    files = os.listdir(origin_path)
    distSum = 0
    for f in files:
        x = read_image(os.path.join(origin_path, f)).to(device)
        x = x / 255.
        y = read_image(os.path.join(gene_path, f)).to(device)
        y = y / 255.
        x.unsqueeze_(0)
        y.unsqueeze_(0)
        dists_value = D(x, y)
        distSum += dists_value.item()
    del D

    return distSum / len(files)


def eval_fid(origin_path, gene_path):
    fid_value = fid_score.calculate_fid_given_paths([origin_path, gene_path],
                                                    batch_size=24, device='cuda:0', dims=2048)
    return fid_value


def eval_lpips(origin_path, gene_path):
    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    loss_fn.cuda()
    files = os.listdir(origin_path)
    i = 0
    total_lpips_distance = 0

    for file in files:

        try:
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(origin_path, file))).cuda()
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(gene_path, file))).cuda()

            if os.path.exists(os.path.join(origin_path, file)) and os.path.exists(os.path.join(gene_path, file)):
                i = i + 1

            # Compute distance
            current_lpips_distance = loss_fn.forward(img0, img1)
            total_lpips_distance = total_lpips_distance + current_lpips_distance.item()

            # print('%s: %.3f'%(file, current_lpips_distance))

        except Exception as e:
            print(e)

    del loss_fn
    return total_lpips_distance / i


@torch.no_grad()
def inference(model, x, recon_dir, name=None):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    x_padded = x_padded / 255.
    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    out_dec["x_hat"] = out_dec["x_hat"] * 255.
    if name:
        out = out_dec["x_hat"].clone().squeeze(0)
        out = out.permute(1, 2, 0)
        out = out.cpu().numpy()

        # print(out.shape)
        I_ll = out.astype(np.uint8)
        im_nll = Image.fromarray(I_ll)
        im_nll.save(recon_dir + '/' + name[-11:])

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    return {
        "psnr": metrics["psnr-rgb"],
        "ms-ssim": metrics["ms-ssim"],
        "log_ssim": metrics["log_ssim"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    x_padded = x_padded / 255.
    start = time.time()
    # print(x_padded.shape)
    out_net = model.forward(x_padded)
    elapsed_time = time.time() - start

    out_net["x_hat"] = F.pad(
        out_net["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    out_net["x_hat"] = out_net["x_hat"] * 255.
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr": metrics["psnr-rgb"],
        "ms-ssim": metrics["ms-ssim"],
        "log_ssim": metrics["log_ssim"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def eval_model(model, filepaths, recon_dir, entropy_estimation=False, half=False):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in filepaths:
        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.to(torch.bfloat16)
                x = x.to(torch.bfloat16)
            rv = inference(model, x, recon_dir, f)
        else:
            rv = inference_entropy_estimation(model, x)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics
