import random

import torch
from kornia.augmentation import RandomCrop
from torchvision.transforms import ToTensor, Compose

from datasets import Vimeo90k
from metrics import psnr, ssim
from models.vsrvc import load_model
from PIL import Image
from utils import save_video, dict_to_string


def generate_video(chkpt_path: str, keyframe1_paths: list, keyframe2_paths: list, save_root: str = None):
    model = load_model(chkpt_path)
    model.eval()
    transform = ToTensor()
    crop = RandomCrop((128, 192), same_on_batch=True)
    for path1, path2 in zip(keyframe1_paths, keyframe2_paths):
        keyframe1 = transform(Image.open(path1).convert("RGB")).to(model.device)
        keyframe2 = transform(Image.open(path2).convert("RGB")).to(model.device)
        cropped = crop(torch.stack([keyframe1, keyframe2]))
        video = model.generate_video_from_2_frames(cropped[0].unsqueeze(0), cropped[1].unsqueeze(0))
        if save_root is not None:
            save_video(video[0], save_root, "generated_video_from_2_keyframes")


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
        generate_video_from_2_keyframes(chkpt, [r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\008\00000000.png"],
                                        [r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\008\00000001.png"],
                                        save_root="../outputs/1713890931.386187")
        generate_video(chkpt, [r"D:\Code\basicvsr\BasicVSR_PlusPlus\data\REDS\train_sharp\008\00000000.png"],
                       save_root="../outputs/1713890931.386187")
        evaluate_model_e2e(chkpt, "../../Datasets/VIMEO90k", 2, [791], save_root="../outputs/1713890931.386187")
