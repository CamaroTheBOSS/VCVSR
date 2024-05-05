import os
import subprocess

import numpy as np
import torch
from torch import Tensor
from torchvision.io import read_video

from metrics import psnr, ssim


def run_compression(yuv_path: str, crf_val: int, codec: str = "hevc", keyframe_interval: int = 0, quite: bool = False,
                    save_root: str = "./", max_frames: int = 0):
    codecs = {"hevc": "libx265", "avc": "libx264", "h263": "h263", "h262": "mpeg2video", "h261": "h261"}
    ffreport = f"{save_root}/ffreport_{crf_val}.log"
    os.environ["FFREPORT"] = f"file={ffreport}:level=56"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if os.path.exists(ffreport):
        os.remove(ffreport)
    output_path = f"{save_root}/{codec}_{crf_val}.mkv"
    keyframe_args = ["-g", str(keyframe_interval)] if keyframe_interval > 0 else []
    max_frames_args = ["-vframes", str(max_frames)] if max_frames > 0 else []

    command = ["ffmpeg",
               "-y",
               "-pix_fmt", "yuv420p",
               "-s", "1920x1080",
               "-i", f"{yuv_path}",
               "-vf", "crop=1920:1024:0:0,scale=960:512",
               *max_frames_args,
               "-c:v", codecs[codec],
               "-crf", str(crf_val),
               *keyframe_args,
               output_path
               ]
    kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if quite else {}
    subprocess.run(command, **kwargs)
    return output_path, ffreport


def get_bpp_from_ffmpeg(ffmpeg_report: str, video_resolution: tuple):
    with open(ffmpeg_report) as f:
        content = f.readlines()

    frame_sizes = []
    for line in content:
        if "of size " in line:
            splitted = line.split(" ")
            size = splitted[7]
            frame_sizes.append(int(size))
    nbits_per_frame = np.array(frame_sizes) * 8.0 / (video_resolution[0] * video_resolution[1])
    return nbits_per_frame.mean()


@torch.no_grad()
def get_psnr_ssim_video_input(original: Tensor, mkv_path: str, nframes: int = 100, aggregate="mean"):
    target, _, _ = read_video(mkv_path)
    indexes = torch.linspace(0, original.shape[1] - 1, nframes).int().to(original.device)

    target = torch.index_select(target.to(original.device), 0, indexes)
    original = torch.index_select(original, 1, indexes)
    target = target.permute(0, 3, 1, 2).unsqueeze(0).float() / 255.

    psnr_value = psnr(target, original, aggregate=aggregate)
    ssim_value = ssim(target, original, aggregate=aggregate)
    return psnr_value.detach().cpu().tolist(), ssim_value.detach().cpu().tolist(),


@torch.no_grad()
def get_psnr_ssim_tensor_input(original: Tensor, target: Tensor, nframes: int = 100, aggregate="mean"):
    indexes = torch.linspace(0, original.shape[1] - 1, nframes).int().to(original.device)
    target = torch.index_select(target.to(original.device), 1, indexes)
    original = torch.index_select(original, 1, indexes)
    psnr_value = psnr(target, original, aggregate=aggregate)
    ssim_value = ssim(target, original, aggregate=aggregate)
    return psnr_value.detach().cpu().tolist(), ssim_value.detach().cpu().tolist(),
