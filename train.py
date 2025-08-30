import argparse
import math
import numpy as np
import os
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from PIL import Image
from torch.optim.optimizer import Args
from models import models
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from accelerate import Accelerator

warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = True


def configure_optimizers(model, args):
    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )

    return optimizer, aux_optimizer


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]

        return out


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, train_step,
                    accelerator, tb_writer=None, clip_max_norm=None):
    model.train()
    
    train_size = 0
    for x in train_dataloader:
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = model(x)
        out_criterion = criterion(out, x)

        accelerator.backward(out_criterion["loss"])

        if clip_max_norm:
            accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)
            
        optimizer.step()
        
        aux_loss = model.aux_loss()

        # 使用 accelerator.backward
        accelerator.backward(aux_loss)
        aux_optimizer.step()

        train_step += 1
        if accelerator.is_main_process and tb_writer and train_step % 10 == 1:
            tb_writer.add_scalar('train loss', out_criterion["loss"].item(), train_step)
            tb_writer.add_scalar('train mse', out_criterion["mse_loss"].item(), train_step)
            tb_writer.add_scalar('train img bpp', out_criterion["bpp_loss"].item(), train_step)
            tb_writer.add_scalar('train aux', aux_loss.item(), train_step)

        train_size += x.shape[0]

    if accelerator.is_main_process:
        print("train sz:{}".format(train_size * accelerator.num_processes)) 
    return train_step


def eval_epoch(model, criterion, eval_dataloader, epoch, accelerator, tb_writer=None):
    model.eval()

    loss = 0
    img_bpp = 0
    mse_loss = 0
    aux_loss_val = 0
    eval_size = 0
    
    save_imgs = tb_writer is not None

    with torch.no_grad():
        for x in eval_dataloader:
            out = model(x)
            out_criterion = criterion(out, x)
            N = x.shape[0]

            loss += out_criterion["loss"] * N
            img_bpp += out_criterion["bpp_loss"] * N
            mse_loss += out_criterion["mse_loss"] * N
            aux_loss_val += model.aux_loss() * N
            eval_size += N
            
            if accelerator.is_main_process and save_imgs:
                x_rec = (out["x_hat"].float() * 255.).clamp_(0, 255)
                x_orig = x.float() * 255.
                tb_writer.add_image('input/0', x_orig[0, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('input/1', x_orig[1, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('input/2', x_orig[2, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('output/0', x_rec[0, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('output/1', x_rec[1, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('output/2', x_rec[2, :, :, :].to(torch.uint8), epoch)
                save_imgs = False
    
    total_loss = accelerator.gather(loss.unsqueeze(0)).sum()
    total_img_bpp = accelerator.gather(img_bpp.unsqueeze(0)).sum()
    total_mse_loss = accelerator.gather(mse_loss.unsqueeze(0)).sum()
    total_aux_loss_val = accelerator.gather(aux_loss_val.unsqueeze(0)).sum()
    total_eval_size = accelerator.gather(torch.tensor([eval_size], device=accelerator.device)).sum()

    if accelerator.is_main_process:
        final_loss = (total_loss / total_eval_size).item()
        final_img_bpp = (total_img_bpp / total_eval_size).item()
        final_mse_loss = (total_mse_loss / total_eval_size).item()
        final_aux_loss = (total_aux_loss_val / total_eval_size).item()
        psnr = 10. * np.log10(1. ** 2 / final_mse_loss)
        
        if tb_writer:
            tb_writer.add_scalar('eval/eval loss', final_loss, epoch)
            tb_writer.add_scalar('eval/eval img bpp', final_img_bpp, epoch)
            tb_writer.add_scalar('eval/eval mse', final_mse_loss, epoch)
            tb_writer.add_scalar('eval/eval psnr', psnr, epoch)
            tb_writer.add_scalar('eval/eval aux', final_aux_loss, epoch)
        
        print("eval sz:{}".format(total_eval_size.item()))
        print("Epoch(Eval):{}, img bpp:{}, mse:{}, psnr:{}".format(epoch, final_img_bpp, final_mse_loss, psnr))
        
        return final_loss, final_img_bpp, final_mse_loss, psnr, final_aux_loss

    return None, None, None, None, None

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4] + "_best" + filename[-4:])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-e",
        "--epochs",
        default=40,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=16,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0067,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=0.0067,
        choices=['factorized', 'hyperprior', 'mbt2018-mean', 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn],
        help="Used Model (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Eval batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--M",
        type=int,
        default=192,
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda (ignored when using accelerator)")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--half", action="store_true", default=False, help="Use FP16 mixed precision training"
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="Use BF16 mixed precision training"
    )
    parser.add_argument(
        "--save_path", type=str, default="./ckp_ll", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", default=0, type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--log_dir", default="./logs_ll/", type=str, help="Path to save log")
    parser.add_argument("--recon_dir", default="./test", type=str, help="Test reconstruction image path"
                        )
    parser.set_defaults(cuda=True)
    args = parser.parse_args(argv)
    return args


class ImageDataset(data.Dataset):
    def __init__(self, path_dir, img_mode=None, transform=None):
        self.path_dir = path_dir
        self.img_mode = img_mode
        self.transform = transform
        self.images = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.path_dir, self.images[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img


def main(argv):
    args = parse_args(argv)
    print(args)

    mixed_precision = "no"
    if args.half and args.bf16:
        raise ValueError("Cannot use --half (fp16) and --bf16 together.")
    elif args.half:
        mixed_precision = "fp16"
    elif args.bf16:
        mixed_precision = "bf16"
    
    accelerator = Accelerator(mixed_precision=mixed_precision)
    accelerator.print(f"Using {accelerator.num_processes} GPUs for training with {mixed_precision} precision.")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if accelerator.is_main_process:
        if args.save_path and not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    eval_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageDataset("your/train/data", transform=train_transforms)
    eval_dataset = ImageDataset("your/test/data", transform=eval_transforms)

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    eval_dataloader = data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    net = models[args.model](N=args.N, M=args.M)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[36], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    
    tb_writer = SummaryWriter(args.log_dir) if accelerator.is_main_process else None
    train_step = 0

    net, optimizer, aux_optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        net, optimizer, aux_optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    last_epoch = 0
    if args.checkpoint:
        accelerator.print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        unwrapped_net = accelerator.unwrap_model(net)
        unwrapped_net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        train_step = checkpoint["step"]

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        if accelerator.is_main_process:
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        train_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            train_step,
            accelerator,
            tb_writer,
            args.clip_max_norm,
        )

        loss, *_ = eval_epoch(net, criterion, eval_dataloader, epoch, accelerator, tb_writer)

        lr_scheduler.step()

        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                unwrapped_net = accelerator.unwrap_model(net)
                checkpoint_dict = {
                    "epoch": epoch,
                    "state_dict": unwrapped_net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "step": train_step
                }
                save_checkpoint(
                    checkpoint_dict,
                    is_best,
                    os.path.join(args.save_path, f"ckp{int(args.lmbda * 10000)}.tar"),
                )
            print("---------------")
            
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])