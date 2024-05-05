import glob
import os
import pathlib
import random

import torch
from kornia.augmentation import RandomCrop, Resize
from torch import nn
from torchvision.transforms import ToTensor

from datasets import Vimeo90k, UVGDataset
from metrics import psnr, ssim
from models.vsrvc import load_model
from PIL import Image
from utils import save_video, dict_to_string, save_frame
import matplotlib.pyplot as plt


def sample(cdf, x):
    probs = torch.rand(x.size()).to(x.device)
    dist_indices = torch.argmin(torch.abs(probs.unsqueeze(-1) - cdf), dim=-1)
    values = torch.gather(x.repeat(len(cdf), 1), 1, dist_indices)
    return values


@torch.no_grad()
def draw_model_distributions(model: nn.Module):
    x = torch.linspace(-25, 25, 3000).to(model.device)
    outputs = model.residual_compressor.bit_estimator.distribution.cdf(x)
    sampled = sample(outputs[0, :], x)

    for idx in range(128):
        hg = torch.histogram(sampled[idx].detach().cpu(), bins=20, density=True)
        hist = (hg.hist - hg.hist.min()) / (hg.hist.max() - hg.hist.min())
        bins = hg.bin_edges[1:]
        plt.plot(bins.detach().numpy(), hist.numpy())
        plt.plot(x.detach().cpu().numpy(), outputs[0, idx, 0].detach().cpu().numpy())
        plt.show()


@torch.no_grad()
def eval_generation(model: nn.Module, keyframe1_paths: list, keyframe2_paths: list, save_root: str = None):
    transform = ToTensor()
    crop = RandomCrop((128, 192), same_on_batch=True)
    for path1, path2 in zip(keyframe1_paths, keyframe2_paths):
        keyframe1 = transform(Image.open(path1).convert("RGB")).to(model.device)
        keyframe2 = transform(Image.open(path2).convert("RGB")).to(model.device)
        cropped = crop(torch.stack([keyframe1, keyframe2]))
        video = model.generate_video(cropped[0].unsqueeze(0), cropped[1].unsqueeze(0))
        if save_root is not None:
            save_video(video[0], save_root, "generated_video_from_2_keyframes")


@torch.no_grad()
def eval_replaced_keyframe(model: nn.Module, dataset, another_keyframe: str,
                           examples: list = None, save_root: str = None):
    if examples is None:
        examples = [random.randint(0, len(dataset))]
    for example in examples:
        lqs, hqs = dataset.__getitem__(example)
        lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
        data_root, bpp, upscaled = model.compress(lqs, save_root=save_root, keyframe_format="png")
        reconstructed_original_keyframe = model.decompress(data_root)

        keyframe = Image.open(another_keyframe).convert("RGB")

        h, w = lqs.shape[-2:]
        y1, x1 = torch.randint(0, keyframe.height - h, ()).item(), torch.randint(0, keyframe.width - w, ()).item()
        keyframe = keyframe.crop((x1, y1, x1 + w, y1 + h))
        keyframe.save(os.path.join(data_root, "keyframe.png"))
        reconstructed_another_keyframe = model.decompress(data_root)

        if save_root is not None:
            save_video(reconstructed_original_keyframe[0], save_root, "keyframe_original")
            save_video(reconstructed_another_keyframe[0], save_root, "keyframe_different")


@torch.no_grad()
def eval_compression(model: nn.Module, dataset, examples: list = None,
                     keyframe_format: str = "png", save_root: str = None):
    if examples is None:
        examples = [random.randint(0, len(dataset))]
    if keyframe_format not in ["png", "jpg"]:
        raise NotImplementedError("Supported formats: png, jpg")
    sf = 1 if keyframe_format == "png" else 0

    for example in examples:
        lqs, hqs = dataset.__getitem__(example)
        lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
        data_root, bpp, upscaled = model.compress(lqs, save_root=save_root, keyframe_format=keyframe_format)
        reconstructed = model.decompress(data_root)
        results = {"vc_psnr": psnr(reconstructed[0, sf:], lqs[0, sf:], aggregate="none"),
                   "vc_ssim": ssim(reconstructed[0, sf:], lqs[0, sf:], aggregate="none"),
                   "vsr_psnr": psnr(upscaled[0], hqs[0], aggregate="none"),
                   "vsr_ssim": ssim(upscaled[0], hqs[0], aggregate="none"),
                   "bpp": bpp}
        delimiter = "\n   "
        print(f"[{example}]:{delimiter}{dict_to_string(results, delimiter=delimiter)}")

        if save_root is not None:
            save_video(reconstructed[0], save_root, "evaluate_compression")
            save_video(upscaled[0], save_root, "evaluate_upscale")


@torch.no_grad()
def eval_upscale_consistency(model: nn.Module, path_to_video: str, save_root="./"):
    video = []
    paths = glob.glob(f"{path_to_video}/*.png")
    transform = ToTensor()
    for path in paths:
        video.append(transform(Image.open(path).convert("RGB")))
    hqs = torch.stack(video)[:, :, :640]
    n, c, h, w = hqs.shape
    lqs = Resize((int(h / 2), int(w / 2)))(hqs)
    lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
    data_root, bpp, upscaled = model.compress(lqs, save_root=save_root, keyframe_format="png")

    col = torch.randint(0, upscaled.shape[-1] - 1, ()).item()
    original = hqs[0, :, :, :, col].permute(1, 2, 0)
    prediction = upscaled[0, :, :, :, col].permute(1, 2, 0)
    save_frame(f"{save_root}/consistency_original.png", original)
    save_frame(f"{save_root}/consistency_prediction{col}.png", prediction)
    save_video(upscaled[0], save_root, "consistency_video")


@torch.no_grad()
def eval_estimated_bits(model: nn.Module, dataset, examples: list = None, save_root="./"):
    if examples is None:
        examples = [random.randint(0, len(dataset))]

    for example in examples:
        lqs, hqs = dataset.__getitem__(example)
        lqs = lqs.to(model.device).unsqueeze(0)
        hqs = hqs.to(model.device).unsqueeze(0)
        bpps = []
        previous_frames = lqs[:, 0]
        for i in range(1, lqs.shape[1]):
            outputs = model(previous_frames, lqs[:, i], hqs[:, i])
            previous_frames = outputs.reconstructed
            bpps.append(sum([value.item() if "bits" in key else 0 for key, value in outputs.loss_vc.items()]))
        _, real_bpp, _ = model.compress(lqs, save_root=save_root, keyframe_format="none")
        estimated_bpp = sum(bpps) / len(bpps)
        print(estimated_bpp, real_bpp)


if __name__ == "__main__":
    chkpt = "../outputs/baseline1024/model_30.pth"
    model = load_model(chkpt)
    model.eval()
    save_root = str(pathlib.Path(chkpt).parent)

    # vimeo = Vimeo90k("../../Datasets/VIMEO90k", 2, test_mode=True)
    uvg = UVGDataset("../../Datasets/UVG", 2, max_frames=100)

    first_keyframe = r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\008\00000000.png"
    second_keyframe = r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\008\00000001.png"
    reds = r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\174"

    # eval_estimated_bits(model, uvg, [0], save_root=save_root)
    # eval_upscale_consistency(model, reds, save_root=save_root)
    eval_compression(model, uvg, [6], save_root=save_root, keyframe_format="jpg")
    # eval_generation(model, [first_keyframe], [second_keyframe], save_root=save_root)
    # eval_replaced_keyframe(model, uvg, r"D:\Code\Datasets\UVG\Jockey\001.png", [0], save_root=save_root)
