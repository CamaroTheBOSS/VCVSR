import glob
import os
import pathlib
import random

import torch
from kornia.augmentation import RandomCrop, Resize
from torchvision.transforms import ToTensor

from datasets import Vimeo90k
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


def draw_model_distributions(chkpt_path: str):
    model = load_model(chkpt_path)
    model.eval()

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


def generate_video(chkpt_path: str, keyframe1_paths: list, keyframe2_paths: list, save_root: str = None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    model = load_model(chkpt_path)
    model.eval()
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
def decompress_video_different_keyframe(chkpt_path: str, dataset_path: str, scale: int, another_keyframe: str,
                                        examples: list = None, save_root: str = None):
    dataset = Vimeo90k(dataset_path, scale, test_mode=True)
    model = load_model(chkpt_path)
    model.eval()
    if examples is None:
        examples = [random.randint(0, len(dataset))]
    for example in examples:
        lqs, hqs = dataset.__getitem__(example)
        lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
        data_root, bpp, upscaled = model.compress(lqs, keyframe_format="png")
        reconstructed_original_keyframe = model.decompress(data_root)

        keyframe = Image.open(another_keyframe).convert("RGB")

        y1, x1 = torch.randint(0, keyframe.height - 128, ()).item(), torch.randint(0, keyframe.width - 192, ()).item()
        keyframe = keyframe.crop((x1, y1, x1 + 192, y1 + 128))
        keyframe.save(os.path.join(data_root, "keyframe.png"))
        reconstructed_another_keyframe = model.decompress(data_root)

        if save_root is not None:
            save_video(reconstructed_original_keyframe[0], save_root, "keyframe_original")
            save_video(reconstructed_another_keyframe[0], save_root, "keyframe_different")


@torch.no_grad()
def evaluate_model_e2e(chkpt_path: str, dataset_path: str, scale: int, examples: list = None,
                       keyframe_format: str = "png", save_root: str = None):
    dataset = Vimeo90k(dataset_path, scale, test_mode=True)
    model = load_model(chkpt_path)
    model.eval()
    if examples is None:
        examples = [random.randint(0, len(dataset))]
    if keyframe_format not in ["png", "jpg"]:
        raise NotImplementedError("Supported formats: png, jpg")
    sf = 1 if keyframe_format == "png" else 0

    for example in examples:
        lqs, hqs = dataset.__getitem__(example)
        lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
        data_root, bpp, upscaled = model.compress(lqs, keyframe_format=keyframe_format)
        reconstructed = model.decompress(data_root)
        results = {"vc_psnr": psnr(reconstructed[0, sf:], lqs[0, sf:], aggregate="none"),
                   "vc_ssim": ssim(reconstructed[0, sf:], lqs[0, sf:], aggregate="none"),
                   "vsr_psnr": psnr(upscaled[0], hqs[0], aggregate="none"),
                   "vsr_ssim": ssim(upscaled[0], hqs[0], aggregate="none"),
                   "bpp": bpp}
        delimiter = "\n   "
        print(f"[{example}]:{delimiter}{dict_to_string(results, delimiter=delimiter)}")

        if save_root is not None:
            save_video(reconstructed[0], save_root, "evaluate_model_e2e_recon")
            save_video(upscaled[0], save_root, "evaluate_model_e2e_upscale")


@torch.no_grad()
def evaluate_upscaled_consistency(chkpt_path: str, path_to_video: str):
    model = load_model(chkpt_path)
    model.eval()

    video = []
    paths = glob.glob(f"{path_to_video}/*.png")
    transform = ToTensor()
    for path in paths:
        video.append(transform(Image.open(path).convert("RGB")))
    hqs = torch.stack(video)[:, :, :640]
    n, c, h, w = hqs.shape
    lqs = Resize((int(h / 2), int(w / 2)))(hqs)
    lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
    data_root, bpp, upscaled = model.compress(lqs, keyframe_format="png")

    col = torch.randint(0, upscaled.shape[-1] - 1, ()).item()
    original = hqs[0, :, :, :, col].permute(1, 2, 0)
    prediction = upscaled[0, :, :, :, col].permute(1, 2, 0)
    save_frame("original.png", original)
    save_frame(f"prediction{col}.png", prediction)
    save_video(upscaled[0], ".", "reds")



if __name__ == "__main__":
    for chkpt in ["../outputs/baseline2048/model_30.pth"]:
        save_root = str(pathlib.Path(chkpt).parent)
        first_keyframe = r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\008\00000000.png"
        second_keyframe = r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\008\00000001.png"
        evaluate_upscaled_consistency(chkpt, r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\174")
        # draw_model_distributions(chkpt)
        # evaluate_model_e2e(chkpt, "../../Datasets/VIMEO90k", 2, [791], save_root=save_root, keyframe_format="jpg")
        # generate_video(chkpt, [first_keyframe], [second_keyframe], save_root=save_root)
        # decompress_video_different_keyframe(chkpt, "../../Datasets/VIMEO90k", 2, first_keyframe, [436],
        #                                     save_root=save_root)
