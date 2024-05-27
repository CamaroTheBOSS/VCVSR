import os
import sys
import traceback

import torch
import wandb
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from datasets import Vimeo90k
from models.vsrvc import load_model
from utils import init_run_dir, MetricLogger, add_dict, FileLogger, log_to_wandb, save_video, save_checkpoint, \
    MyCosineAnnealingLR
from metrics import psnr, ssim


@torch.no_grad()
def evaluate_model(dataloader: DataLoader, model: nn.Module, output_dir: str):
    model.eval()
    metric_dict = {"vc_psnr": 0, "vc_ssim": 0, "vsr_psnr": 0, "vsr_ssim": 0, "vc_bpp": 0, "vc_nonzeros_offsets": 0,
                   "vc_nonzeros_residuals": 0}
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            lqs = batch[0].to(model.device)
            hqs = batch[1].to(model.device)

            reconstructed, upscaled, bpps, nonzeros_offsets, nonzeros_residuals = [], [], [], [], []
            previous_frames = lqs[:, 0]
            for i in range(1, lqs.shape[1]):
                outputs = model(previous_frames, lqs[:, i], hqs[:, i])
                reconstructed.append(outputs.reconstructed)
                upscaled.append(outputs.upscaled)
                previous_frames = outputs.reconstructed
                bpps.append(sum([value.item() if "bits" in key else 0 for key, value in outputs.loss_vc.items()]))
                nonzeros_offsets.append(outputs.additional_info["count_non_zeros_offsets"])
                nonzeros_residuals.append(outputs.additional_info["count_non_zeros_residuals"])

            metric_dict["vc_psnr"] += psnr(torch.stack(reconstructed).permute(1, 0, 2, 3, 4), lqs[:, 1:]).item()
            metric_dict["vc_ssim"] += ssim(torch.stack(reconstructed).permute(1, 0, 2, 3, 4), lqs[:, 1:]).item()
            metric_dict["vsr_psnr"] += psnr(torch.stack(upscaled).permute(1, 0, 2, 3, 4), hqs[:, 1:]).item()
            metric_dict["vsr_ssim"] += ssim(torch.stack(upscaled).permute(1, 0, 2, 3, 4), hqs[:, 1:]).item()
            metric_dict["vc_bpp"] += sum(bpps) / len(bpps)
            metric_dict["vc_nonzeros_offsets"] += sum(nonzeros_offsets) * len(dataloader)
            metric_dict["vc_nonzeros_residuals"] += sum(nonzeros_residuals) * len(dataloader)

    save_video(torch.stack(reconstructed).permute(1, 0, 2, 3, 4)[0], f"{output_dir}/compressed")
    save_video(torch.stack(upscaled).permute(1, 0, 2, 3, 4)[0], f"{output_dir}/upscaled")
    metric_dict = {key: value / len(dataloader) for key, value in metric_dict.items()}
    return metric_dict


def train_epoch(dataloader: DataLoader, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, epoch: int):
    model.train()
    metric_logger = MetricLogger("   ")
    loss_wandb = {"epoch": 0}
    for idx, batch in enumerate(metric_logger(dataloader, header=f"Epoch [{epoch}]", print_every=10)):
        if dataloader.dataset.augment:
            video = batch.to(model.device)
            hqs = torch.stack([dataloader.dataset.augmentation(vid) for vid in video])
            lqs = torch.stack([dataloader.dataset.resize(vid) for vid in hqs])
        else:
            lqs = batch[0].to(model.device)
            hqs = batch[1].to(model.device)
        outputs = model(lqs[:, 0], lqs[:, 1], hqs[:, 1])
        loss_vc = sum(loss_value for loss_value in outputs.loss_vc.values())
        loss_vsr = sum(loss_value for loss_value in outputs.loss_vsr.values())
        loss_shared = sum(loss_value for loss_value in outputs.loss_shared.values())
        loss = loss_shared + (loss_vc if model.vc else 0) + (loss_vsr if model.vsr else 0)

        loss_wandb = add_dict(add_dict(add_dict(loss_wandb, outputs.loss_vc), outputs.loss_vsr), outputs.loss_shared)
        loss_wandb["epoch"] += loss.item()

        metric_logger.update({"vc": loss_vc, "vsr": loss_vsr, "shared": loss_shared}, prefix="loss_")
        metric_logger.update({"lr": optimizer.param_groups[0]['lr']})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    loss_wandb = {key: value / len(dataloader) for key, value in loss_wandb.items()}
    return loss_wandb


def main(rdr):
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)
    batch = 16
    scale = 2
    lr = 1e-4
    epochs = 30
    checkpoint = 5
    rate_distortion = rdr
    vc = True
    vsr = True
    checkpoint_path = None
    wandb_enabled = False
    augment = True
    run_name = f"TEST105 {rate_distortion}"
    run_description = f"Rate distortion ratio has impact on super-resolution"
    if wandb_enabled:
        wandb.init(project="VSRVC", name=run_name)

    output_dir = init_run_dir("outputs", run_name)
    logger = FileLogger(output_dir, "train.log")
    logger.log(f"batch: {batch}, scale: {scale}, lr: {lr}, epochs: {epochs}, checkpoint: {checkpoint}, "
               f"checkpoint_path: {checkpoint_path}, wandb: {wandb_enabled}, run_name: {run_name}"
               f"rate_distortion_ratio: {rate_distortion}\n\nRUN DESCRIPTION: \n{run_description}\n")

    train_set = Vimeo90k("../Datasets/VIMEO90k", scale, augment=augment)
    test_set = Vimeo90k("../Datasets/VIMEO90k", scale, test_mode=True, augment=augment)
    train_dataloader = DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_set, batch_size=batch, shuffle=False, drop_last=True)

    model = load_model(checkpoint_path, run_name, rate_distortion, vc, vsr)
    logger.log(model.summary())
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = MyCosineAnnealingLR(optimizer, epochs * len(train_dataloader), 0.01 * lr, starting_point=0.5)

    try:
        for epoch in range(epochs):
            loss_dict = train_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch)
            metric_dict = evaluate_model(test_dataloader, model, f"{output_dir}/epoch_{epoch + 1}")
            print(metric_dict)
            if wandb_enabled:
                log_to_wandb(loss_dict, metric_dict)

            if (epoch + 1) % checkpoint == 0:
                kwargs = {"model_name": run_name, "rate_distortion_ratio": model.rdr, "vc": model.vc, "vsr": model.vsr}
                save_checkpoint(model, output_dir, epoch, only_last=True, **kwargs)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        _, exc_value, tb = sys.exc_info()
        logger.log(traceback.TracebackException(type(exc_value), exc_value, tb, limit=None).format(chain=True))

    if wandb_enabled:
        wandb.finish()


if __name__ == '__main__':
    for r in [2048, 1024, 512, 128]:
        main(r)
