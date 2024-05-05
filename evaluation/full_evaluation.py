import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from operator import itemgetter
from datasets import UVGDataset
from evaluation.ffmpeg import get_psnr_ssim_tensor_input
from models.vsrvc import VSRVCModel, load_model
from prepare_uvg import get_keywords

"""
File where every aspect of the model is evaluated on full UVG dataset without any flexible configurations. 
Just full test.
"""


def plot_superresolution_table(model_databases: List[str], ffmpeg_database: str, quality_metric: str = "psnr"):
    with open(ffmpeg_database, "r") as f:
        ffmpeg_data = json.load(f)

    names = []
    plot_data = {"bilinear": []}
    for data_entry in ffmpeg_data["data"]:
        names.append(data_entry["name"])
        plot_data["bilinear"].append(round(data_entry[f"bilinear_{quality_metric}"], 3))

    for database_path in model_databases:
        with open(database_path, "r") as f:
            database = json.load(f)
        model_quality = []
        for data_entry in database["data"]:
            model_quality.append(round(data_entry[f"vsr_{quality_metric}"], 3))
        plot_data[database["model_name"]] = model_quality

    df = pd.DataFrame(list(zip(["model"] + names, *[[key] + values for key, values in plot_data.items()])))
    to_print = "{0:20}   ".format("model") + "".join(["   {0:20}".format(name) for name in names]) + "\n"
    for key, values in plot_data.items():
        print_values = ["   {0:20}".format(str(value)) for value in values]
        to_print += "{0:20}   ".format(key) + "".join(print_values) + "\n"

    print(quality_metric.upper() + "".join(["-" for i in range(int(len(to_print) / (len(plot_data.keys()) + 1)))]))
    print(to_print)
    df.to_csv(f"../outputs/super-resolution-{quality_metric}.csv")


def plot_compression_curve(model_databases: List[str], ffmpeg_database: str, quality_metric: str = "psnr"):
    with open(ffmpeg_database, "r") as f:
        ffmpeg_data = json.load(f)

    plot_data = {}
    for codec in ["avc", "hevc"]:
        codec_x, codec_y = [], []
        for i, crf in enumerate(ffmpeg_data["crfs"]):
            mean_quality = []
            mean_bpp = []
            for data_entry in ffmpeg_data["data"]:
                mean_quality.append(data_entry[codec][i][quality_metric])
                mean_bpp.append(data_entry[codec][i]["bpp"])
            codec_y.append(sum(mean_quality) / len(mean_quality))
            codec_x.append(sum(mean_bpp) / len(mean_bpp))
        plot_data[codec] = (np.array(codec_x), np.array(codec_y))

    rdrs = []
    for database_path in model_databases:
        with open(database_path, "r") as f:
            database = json.load(f)
        assert ("model_name" in database.keys())

        rdrs.append(database["rdr"])
        mean_quality = []
        mean_bpp = []
        for data_entry in database["data"]:
            mean_quality.append(data_entry[f"vc_{quality_metric}"])
            mean_bpp.append(data_entry["bpp"])
        plot_data[database["model_name"]] = ([sum(mean_bpp) / len(mean_bpp)], [sum(mean_quality) / len(mean_quality)])

    for (x_data, y_data) in plot_data.values():
        if len(x_data) == 1:
            plt.scatter(x_data, y_data)
        else:
            plt.plot(x_data, y_data)
    plt.xlabel("BPP")
    plt.ylabel(quality_metric.upper())
    plt.legend(plot_data.keys())
    plt.title(f"{quality_metric.upper()}-BPP curve for tested models")
    plt.show()


@torch.no_grad()
def full_eval_compression(model: VSRVCModel, model_name: str, dst_path: str, dataset: UVGDataset,
                          keyframe_format: str = "png", save_root=None, n_samples: int = 7):
    if not dst_path.endswith(".json"):
        raise Exception("dst_path must be a valid path to json file")
    if keyframe_format not in ["png", "jpg"]:
        raise NotImplementedError("Supported formats: png, jpg")
    sf = 1 if keyframe_format == "png" else 0
    keywords = get_keywords()

    output = {"model_name": model_name, "max_frames": dataset.max_frames, "rdr": model.rdr, "data": []}
    for keyword in keywords:
        name = keyword[:-1]
        lqs, hqs = dataset.getitem_with_name(name)
        lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
        data_root, bpp, upscaled = model.compress(lqs, save_root=save_root, keyframe_format=keyframe_format,
                                                  verbose=True)
        reconstructed = model.decompress(data_root)
        vc_psnr, vc_ssim = get_psnr_ssim_tensor_input(reconstructed[:, sf:], lqs[:, sf:], nframes=n_samples)
        vsr_psnr, vsr_ssim = get_psnr_ssim_tensor_input(upscaled, hqs, nframes=n_samples)
        results = {"name": name, "bpp": bpp, "vc_psnr": vc_psnr, "vc_ssim": vc_ssim, "vsr_psnr": vsr_psnr,
                   "vsr_ssim": vsr_ssim}
        output["data"].append(results)
        print(f"[{name}]: {results}")

    with open(dst_path, "w") as f:
        json.dump(output, f)
    return dst_path


if __name__ == "__main__":
    checkpoints = ["../outputs/baseline128/model_30.pth", "../outputs/baseline512/model_30.pth",
                   "../outputs/baseline1024/model_30.pth", "../outputs/baseline2048/model_30.pth"]
    # Generate eval databases
    # uvg = UVGDataset("../../Datasets/UVG", 2, max_frames=100)
    # for chkpt in checkpoints:
    #     model = load_model(chkpt)
    #     model.eval()
    #     save_root = str(pathlib.Path(chkpt).parent)
    #     name = save_root.split("\\")[-1]
    #     full_eval_compression(model, name, f"{save_root}/uvg_eval.json", uvg, keyframe_format="jpg", save_root=save_root, n_samples=7)

    # Plot eval databases
    baseline = [f"{str(pathlib.Path(chkpt).parent)}/uvg_eval.json" for chkpt in checkpoints]
    plot_superresolution_table(baseline, "database.json", quality_metric="ssim")
    plot_superresolution_table(baseline, "database.json", quality_metric="psnr")
    # plot_compression_curve(baseline, "database.json", quality_metric="ssim")
    # plot_compression_curve(baseline, "database.json", quality_metric="psnr")
