import random

import torch
from kornia.augmentation import RandomCrop
from torchvision.transforms import ToTensor, Compose

from datasets import Vimeo90k
from metrics import psnr, ssim
from models.vsrvc import load_model
from PIL import Image
from utils import save_video, dict_to_string


def generate_video(chkpt_path: str, keyframe_paths: list, save_root: str = None):
    model = load_model(chkpt_path)
    model.eval()
    transform = Compose([
        ToTensor(),
        RandomCrop((128, 192))
    ])
    for path in keyframe_paths:
        keyframe = transform(Image.open(path).convert("RGB")).to(model.device)
        video = model.generate_video(keyframe)
        if save_root is not None:
            save_video(video[0], save_root, "generated_video_from_normal_dist")


@torch.no_grad()
def evaluate_model_e2e(chkpt_path: str, dataset_path: str, scale: int, examples: list = None, save_root: str = None):
    dataset = Vimeo90k(dataset_path, scale, test_mode=True)
    model = load_model(chkpt_path)
    model.eval()
    if examples is None:
        examples = [random.randint(0, len(dataset))]

    for example in examples:
        lqs, hqs = dataset.__getitem__(example)
        lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
        data_root, bpp, upscaled = model.compress(lqs, keyframe_format="png")
        reconstructed = model.decompress(data_root)
        results = {"vc_psnr": psnr(reconstructed[0, 1:], lqs[0, 1:], aggregate="none"),
                   "vc_ssim": ssim(reconstructed[0, 1:], lqs[0, 1:], aggregate="none"),
                   "vsr_psnr": psnr(upscaled[0], hqs[0], aggregate="none"),
                   "vsr_ssim": ssim(upscaled[0], hqs[0], aggregate="none"),
                   "bpp": bpp}
        delimiter = "\n   "
        print(f"[{example}]:{delimiter}{dict_to_string(results, delimiter=delimiter)}")

        if save_root is not None:
            save_video(reconstructed[0], save_root, "evaluate_model_e2e_recon")
            save_video(upscaled[0], save_root, "evaluate_model_e2e_upscale")


if __name__ == "__main__":
    for chkpt in ["../outputs/1713890931.386187/model_15.pth"]:
        generate_video(chkpt, [r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\008\00000000.png"],
                       save_root="../outputs/1713890931.386187")
        evaluate_model_e2e(chkpt, "../../Datasets/VIMEO90k", 2, [791], save_root="../outputs/1713890931.386187")
