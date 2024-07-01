import glob
import json
import os
import pathlib
import random
import shutil

import numpy as np
import torch
from kornia.augmentation import RandomCrop, Resize
from torch import nn
from torchvision.transforms import ToTensor

from datasets import Vimeo90k, UVGDataset
from metrics import psnr, ssim
from PIL import Image
from utils import save_video, dict_to_string, save_frame
from models.load_model import load_model
import matplotlib.pyplot as plt


def derivative(cdf):
    prev_cdf = cdf[:, 0, :-1]
    next_cdf = cdf[:, 0, 1:]
    derivative_values = next_cdf - prev_cdf
    return derivative_values


@torch.no_grad()
def draw_model_distributions_deriv(model: nn.Module):
    x = torch.linspace(-25, 25, 5000).to(model.device)
    outputs = model.residual_compressor.bit_estimator.distribution.cdf(x)
    derivative_values = derivative(outputs[0, :]).detach().cpu().numpy()

    x = x.detach().cpu().numpy()
    outputs = outputs[0, :, 0].detach().cpu().numpy()
    for idx in range(128):
        deriv = (derivative_values[idx] - derivative_values[idx].min()) / (derivative_values[idx].max() - derivative_values[idx].min())
        outputs[idx][outputs[idx] < 1e-6] = 0
        outputs[idx][outputs[idx] > 1 - 1e-6] = 1
        plt.plot(x, outputs[idx], "-b")
        plt.plot(x[:-1], deriv, "-r")
        plt.xlabel("Wartość")
        plt.ylabel("Prawdopodobieństwo")
        plt.xlim([x[len(outputs[idx]) - np.argmin(np.flip(outputs[idx]))], x[np.argmax(outputs[idx])]])
        plt.legend(["Wytrenowana dystrybuanta", "Rozkład wartości"])
        plt.show()


@torch.no_grad()
def eval_generation(model: nn.Module, dataset, example: int, save_root: str = None):
    lqs, _ = dataset[example]
    video = model.generate_video(lqs[0].unsqueeze(0), lqs[1].unsqueeze(0), nframes=lqs.shape[0])
    if save_root is not None:
        save_video(video[0], save_root, "generated_video/generated")
        save_video(lqs, save_root, "generated_video/original")


@torch.no_grad()
def eval_motion_transfer(model: nn.Module, dataset, first_example: int, second_example: int, save_root: str = "./"):
    compressed_data_paths = []
    for i, example in enumerate([first_example, second_example]):
        print(f"Compressing {i+1}")
        lqs, _ = dataset[example]
        lqs = lqs.to(model.device).unsqueeze(0)
        data_root, _, _ = model.compress(lqs, save_root=save_root, keyframe_format="png", folder_name=f"compressed_{i+1}")
        compressed_data_paths.append(data_root)
    print(f"Replacing motion vectors...")
    tmp_path = os.path.join(save_root, "tmp")
    os.makedirs(tmp_path)
    for file in ["keyframe.png"]:
        shutil.move(os.path.join(compressed_data_paths[0], file), os.path.join(tmp_path, file))
        shutil.move(os.path.join(compressed_data_paths[1], file), os.path.join(compressed_data_paths[0], file))
        shutil.move(os.path.join(tmp_path, file), os.path.join(compressed_data_paths[1], file))
    shutil.rmtree(tmp_path)

    for i, data_path in enumerate(compressed_data_paths):
        print(f"Decompressing {i + 1}")
        reconstructed_first = model.decompress(data_path)
        save_video(reconstructed_first[0], save_root, f"motion_transferred_{i+1}")


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
            with open(os.path.join(save_root, f"example_{example}.json"), "w") as f:
                results = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in results.items()}
                json.dump(results, f)

        return lqs, hqs, reconstructed, upscaled


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


@torch.no_grad()
def test_compression(model: nn.Module, dataset: UVGDataset, save_root=None):
    lqs, hqs = dataset.__getitem__(5)
    lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
    data_root, bpp, upscaled, coded_data = model.compress(lqs, save_root=save_root, return_data=True)
    reconstructed, decoded_data = model.decompress(data_root, return_data=True)
    for key in coded_data.keys():
        if key == "quantization_info":
            continue
        coded = torch.stack(coded_data[key]).squeeze(1)
        decoded = decoded_data[key]
        if not torch.equal(coded, decoded):
            print("Unconsistent compress/decompress data!")
        else:
            print("Consistent")



if __name__ == "__main__":
    # chkpt = "../outputs/baseline_no_aug2048/model_30.pth"
    chkpt = "../outputs/backup/VSRVC NAUG 2048/model_30.pth"
    model = load_model(chkpt)
    model.eval()
    save_root = str(pathlib.Path(chkpt).parent)
    dataset = UVGDataset("../../Datasets/UVG", 2, max_frames=100)

    # draw_model_distributions_deriv(model)
    # eval_estimated_bits(model, uvg, [0], save_root=save_root)
    # eval_upscale_consistency(model, reds, save_root=save_root)
    test_compression(model, dataset, save_root)
    # eval_compression(model, dataset, [5], save_root=save_root, keyframe_format="jpg")
    # eval_generation(model, dataset, 3, save_root=save_root)
    # eval_replaced_keyframe(model, dataset, r"D:\Code\Datasets\UVG\YachtRide\001.png", [3], save_root=save_root)
    # eval_motion_transfer(model, dataset, 6, 3, save_root)
