import json
import os.path
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from datasets import UVGDataset
from evaluation.ffmpeg import get_psnr_ssim_tensor_input
from models.vsrvc import VSRVCModel, load_model
from evaluation.prepare_uvg import get_keywords

"""
File where every aspect of the model is evaluated on full UVG dataset without any flexible configurations. 
Just full test.
"""


def get_line_type(key):
    key_dict = {"hevc": "-",
                "avc": "-",
                128: "-",
                512: "--",
                1024: "-",
                2048: "--",
                }
    return key_dict[key]


def get_color(key):
    # Okabe Ito color pallete
    key_dict = {"hevc": "#D55E00",
                "avc": "#E69F00",
                128: "#56B4E9",
                512: "#0072B2",
                1024: "#009E73",
                2048: "#000000",
                }
    return key_dict[key]


def plot_compression_comparison(first_databases: list, other_databases: list, first_name: str, other_name: str):
    first_data, other_data = {}, {}
    for data_dict, databases in zip([first_data, other_data], [first_databases, other_databases]):
        for database_path in databases:
            with open(database_path, "r") as f:
                database = json.load(f)
            assert ("rdr" in database.keys())
            name = database['rdr']
            mean_psnr = []
            mean_ssim = []
            mean_bpp = []
            for data_entry in database["data"]:
                bpp = data_entry["bpp"]
                mean_bpp.append(sum(bpp))
                psnr_values = data_entry[f"vc_psnr"]
                mean_psnr.append(sum(psnr_values) / len(psnr_values))
                ssim_values = data_entry[f"vc_ssim"]
                mean_ssim.append(sum(ssim_values) / len(ssim_values))

            data_dict[name] = (round(sum(mean_bpp) / len(mean_bpp), 3),
                               (round(sum(mean_psnr) / len(mean_psnr), 3),
                                round(sum(mean_ssim) / len(mean_ssim), 3)))

    first_psnr, first_ssim, first_bpp = [], [], []
    other_psnr, other_ssim, other_bpp = [], [], []
    for key, values in first_data.items():
        first_bpp.append(values[0])
        first_psnr.append(values[1][0])
        first_ssim.append(values[1][1])
    for key, values in other_data.items():
        other_bpp.append(values[0])
        other_psnr.append(values[1][0])
        other_ssim.append(values[1][1])

    plt.scatter(first_bpp, first_psnr, color=get_color(1024))
    plt.scatter(other_bpp, other_psnr, color=get_color(2048))
    plt.xlabel("BPP")
    plt.ylabel("PSNR")
    plt.legend([first_name, other_name])
    plt.show()

    plt.scatter(first_bpp, first_ssim, color=get_color(1024))
    plt.scatter(other_bpp, other_ssim, color=get_color(2048))
    plt.xlabel("BPP")
    plt.ylabel("SSIM")
    plt.legend([first_name, other_name])
    plt.show()


def plot_output_files_size(model_databases: List[str]):
    filenames = ["klatka kluczowa", "wektory ruchu w podprzestrzeni", "wektory ruchu w hiperprzestrzeni",
                 "rezydua w podprzestrzeni", "rezydua w hiperprzestrzeni"]
    plot_data = {}
    for database_path in model_databases:
        with open(database_path, "r") as f:
            database = json.load(f)
        motion_bpp = 0
        residual_bpp = 0
        keyframe_bpp = 0
        name = database['rdr']
        for data_entry in database["data"]:
            for bpp, filename in zip(data_entry["bpp"], filenames):
                keyframe_bpp = keyframe_bpp + bpp if filename.startswith("klatka") else keyframe_bpp
                motion_bpp = motion_bpp + bpp if filename.startswith("wektory") else motion_bpp
                residual_bpp = residual_bpp + bpp if filename.startswith("rezydua") else residual_bpp
        sum_bpp = motion_bpp + residual_bpp + keyframe_bpp
        keyframe_bpp = round(keyframe_bpp * 100 / sum_bpp, 2)
        motion_bpp = round(motion_bpp * 100 / sum_bpp, 2)
        residual_bpp = round(residual_bpp * 100 / sum_bpp, 2)
        plot_data[name] = (keyframe_bpp, motion_bpp, residual_bpp)

    keyframes = []
    motions = []
    residuals = []
    model_names = [f"λ={key}" for key in plot_data.keys()]
    for i, (key, values) in enumerate(plot_data.items()):
        keyframes.append(values[0])
        motions.append(values[1])
        residuals.append(values[2])

    bar_width = 1 / (len(keyframes) + 1)
    br1 = np.arange(len(keyframes))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]

    # Make the plot
    plt.bar(br1, keyframes, color="#56B4E9", width=bar_width,
            edgecolor='grey', label="klatka kluczowa")
    plt.bar(br2, motions, color="#0072B2", width=bar_width,
            edgecolor='grey', label="wektory ruchu")
    plt.bar(br3, residuals, color="#009E73", width=bar_width,
            edgecolor='grey', label="rezydua")

    plt.xlabel('Model')
    plt.ylabel('Rozmiar względny [%]')
    plt.legend()
    plt.xticks([r + bar_width for r in range(4)], labels=model_names)
    plt.show()


def plot_superresolution_table(model_databases: List[str], ffmpeg_database: str):
    with open(ffmpeg_database, "r") as f:
        ffmpeg_data = json.load(f)

    names = []
    plot_data = {"interpolacja dwuliniowa": ([], [])}
    for data_entry in ffmpeg_data["data"]:
        names.append(data_entry["name"])
        for i, metric in enumerate(["psnr", "ssim"]):
            bilinear = data_entry[f"bilinear_{metric}"]
            plot_data["interpolacja dwuliniowa"][i].append(bilinear)
    plot_data["interpolacja dwuliniowa"] = (np.round(np.array(plot_data["interpolacja dwuliniowa"][0]).mean(), 2),
                                            np.round(np.array(plot_data["interpolacja dwuliniowa"][1]).mean(), 2))

    for database_path in model_databases:
        with open(database_path, "r") as f:
            database = json.load(f)
        name = database['rdr']
        plot_data[name] = ([], [])
        for data_entry in database["data"]:
            for i, metric in enumerate(["vsr_psnr", "vsr_ssim"]):
                bilinear = data_entry[metric]
                plot_data[name][i].append(bilinear)
        plot_data[name] = (np.round(np.array(plot_data[name][0]).mean(), 2),
                           np.round(np.array(plot_data[name][1]).mean(), 2))

    df = pd.DataFrame(list(zip(*[[f"$\lambda={key}$", value] for key, value in plot_data.items()])))
    df.to_csv(f"../outputs/super-resolution-quality.csv", index=False, header=False)
    print(df.to_string(index=False, header=False))


def plot_compression_curve(model_databases: List[str], ffmpeg_database: str):
    with open(ffmpeg_database, "r") as f:
        ffmpeg_data = json.load(f)

    codec_plot_data = {}
    for codec in ["avc", "hevc"]:
        codec_x = []
        codec_psnr = []
        codec_ssim = []
        for i, crf in enumerate(ffmpeg_data["crfs"]):
            mean_psnr = []
            mean_ssim = []
            mean_bpp = []
            for data_entry in ffmpeg_data["data"]:
                bpp = data_entry[codec][i]["bpp"]
                mean_bpp.append(sum(bpp) / len(bpp))
                psnr_values = data_entry[codec][i]["psnr"]
                mean_psnr.append(sum(psnr_values) / len(psnr_values))
                ssim_values = data_entry[codec][i]["ssim"]
                mean_ssim.append(sum(ssim_values) / len(ssim_values))
            codec_x.append(sum(mean_bpp) / len(mean_bpp))
            codec_psnr.append(sum(mean_psnr) / len(mean_psnr))
            codec_ssim.append(sum(mean_ssim) / len(mean_ssim))

        codec_plot_data[codec] = (np.flip(np.array(codec_x)),
                                  (np.flip(np.array(codec_psnr)), np.flip(np.array(codec_ssim))))

    rdrs = []
    model_plot_data = {}
    for database_path in model_databases:
        with open(database_path, "r") as f:
            database = json.load(f)
        assert ("rdr" in database.keys())
        name = database['rdr']
        rdrs.append(database["rdr"])
        mean_psnr = []
        mean_ssim = []
        mean_bpp = []
        for data_entry in database["data"]:
            bpp = data_entry["bpp"]
            mean_bpp.append(sum(bpp))
            psnr_values = data_entry[f"vc_psnr"]
            mean_psnr.append(sum(psnr_values) / len(psnr_values))
            ssim_values = data_entry[f"vc_ssim"]
            mean_ssim.append(sum(ssim_values) / len(ssim_values))

        model_plot_data[name] = (round(sum(mean_bpp) / len(mean_bpp), 3),
                                 (round(sum(mean_psnr) / len(mean_psnr), 3),
                                  round(sum(mean_ssim) / len(mean_ssim), 3)))

    to_df = [["BPP", f"model"] + [f"{c.upper()}" for c in codec_plot_data.keys()]]
    for key, (bpp, (psnr, ssim)) in model_plot_data.items():
        print_values = [str(bpp), f"$\lambda={key}$, {(psnr, ssim)}"]
        for codec_bpps, (codec_psnrs, codec_ssims) in codec_plot_data.values():
            interp_psnr = np.interp(np.array([bpp]), codec_bpps, codec_psnrs)
            interp_ssim = np.interp(np.array([bpp]), codec_bpps, codec_ssims)
            entry = (round(interp_psnr[0], 3), round(interp_ssim[0], 3))
            print_values.append(str(entry))
        to_df.append(print_values)

    df = pd.DataFrame(list(zip(*to_df))).transpose()
    df.to_csv(f"../outputs/compression-quality.csv", index=False, header=False)
    print(df.to_string(index=False, header=False))
    for i, metric in enumerate(["PSNR", "SSIM"]):
        for key, (x_data, y_data) in codec_plot_data.items():
            plt.plot(x_data, y_data[i], linestyle=get_line_type(key), color=get_color(key))
        for key, (x_data, y_data) in model_plot_data.items():
            plt.scatter(x_data, y_data[i], color=get_color(key))
        plt.xlabel("BPP")
        plt.ylabel(metric)
        plt.legend(list(codec_plot_data.keys()) + [f"λ={key}" for key in model_plot_data.keys()])
        plt.show()


@torch.no_grad()
def test_uvg(model: VSRVCModel, dst_path: str, dataset: UVGDataset, keyframe_format: str = "jpg", save_root=None):
    model.eval()
    if not dst_path.endswith(".json"):
        raise Exception("dst_path must be a valid path to json file")
    if keyframe_format not in ["png", "jpg"]:
        raise NotImplementedError("Supported formats: png, jpg")
    sf = 1 if keyframe_format == "png" else 0
    keywords = get_keywords()

    output = {"model_name": model.name, "max_frames": dataset.max_frames, "rdr": model.rdr, "data": []}
    for keyword in keywords:
        name = keyword[:-1]
        lqs, hqs = dataset.getitem_with_name(name)
        lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
        data_root, bpp, upscaled = model.compress(lqs, save_root=save_root, keyframe_format=keyframe_format,
                                                  verbose=True, aggregate_bpp="none")
        reconstructed = model.decompress(data_root)
        vc_psnr, vc_ssim = get_psnr_ssim_tensor_input(reconstructed[:, sf:], lqs[:, sf:], aggregate="none")
        vsr_psnr, vsr_ssim = get_psnr_ssim_tensor_input(upscaled, hqs, aggregate="none")
        results = {"name": name, "bpp": bpp, "vc_psnr": vc_psnr, "vc_ssim": vc_ssim, "vsr_psnr": vsr_psnr,
                   "vsr_ssim": vsr_ssim}
        output["data"].append(results)
        print(f"[{name}]: {results}")

    with open(dst_path, "w") as f:
        json.dump(output, f)
    return dst_path


def plot_frame_compression_performance(model_databases: list, quality_metric: str = "psnr"):
    # with open(ffmpeg_database, "r") as f:
    #     ffmpeg_data = json.load(f)
    #
    names = []
    # codec_plot_data = {}
    # crf_index = ffmpeg_data["crfs"].index(crf)
    # for codec in ["avc", "hevc"]:
    #     names.append(codec)
    #     qualities = []
    #     for data_entry in ffmpeg_data["data"]:
    #         qualities.append(data_entry[codec][crf_index][quality_metric])
    #     codec_plot_data[codec] = np.array(qualities).mean(axis=0)

    # Model data
    model_plot_data = {}
    for database_path in model_databases:
        with open(database_path, "r") as f:
            database = json.load(f)
        qualities = []
        name = database['rdr']
        for data_entry in database["data"]:
            key = "vc_" + quality_metric
            qualities.append(data_entry[key])
        model_plot_data[name] = np.array(qualities).mean(axis=0)
        names.append(f"λ={name}")

    # for key, y_data in codec_plot_data.items():
    #     plt.plot(list(range(1, len(y_data) + 1)), y_data, linestyle=get_line_type(key), color=get_color(key), linewidth=2.0)
    for key, y_data in model_plot_data.items():
        plt.plot(list(range(1, len(y_data) + 1)), y_data, linestyle=get_line_type(key), color=get_color(key),
                 linewidth=2.0)
    plt.xlabel("Klatka")
    plt.ylabel(quality_metric.upper())
    plt.legend(names)
    plt.show()


def get_baseline(checkpoints):
    return [f"{str(pathlib.Path(chkpt).parent)}/uvg_eval.json" for chkpt in checkpoints]


if __name__ == "__main__":
    checkpoints1 = ["../outputs/baseline_no_aug128/model_30.pth", "../outputs/baseline_no_aug512/model_30.pth",
                   "../outputs/baseline_no_aug1024/model_30.pth", "../outputs/baseline_no_aug2048/model_30.pth"]
    checkpoints2 = ["../outputs/VCVSR 128/model_30.pth", "../outputs/VCVSR 512/model_30.pth",
                   "../outputs/VCVSR 1024/model_30.pth", "../outputs/VCVSR 2048/model_30.pth"]
    #
    # checkpoints = ["../outputs/Baseline VC 128/model_30.pth", "../outputs/Baseline VC 512/model_30.pth",
    #                "../outputs/Baseline VC 1024/model_30.pth", "../outputs/Baseline VC 2048/model_30.pth",
    #                "../outputs/VSR 128/model_30.pth", "../outputs/VSR 512/model_30.pth",
    #                "../outputs/VSR 1024/model_30.pth", "../outputs/VSR 2048/model_30.pth"
    #                ]
    # plot_frame_results_on_given_video(checkpoints, r"C:\Users\CamaroTheBOSS\Downloads", 25, "YachtRide")
    # Generate eval databases
    checkpoints = ["../outputs/VSRVC AUG 2048/model_30.pth"]
    uvg = UVGDataset("../../Datasets/UVG", 2, max_frames=100)
    for chkpt in checkpoints:
        model = load_model(chkpt)
        model.eval()
        save_root = str(pathlib.Path(chkpt).parent)
        name = save_root.split("\\")[-1]
        test_uvg(model, f"{save_root}/uvg_eval.json", uvg, keyframe_format="jpg", save_root=save_root)

    # Plot eval databases
    baseline = [f"{str(pathlib.Path(chkpt).parent)}/uvg_eval.json" for chkpt in checkpoints]
    plot_compression_curve(baseline, "database2.json")
    # plot_frame_compression_performance(baseline, quality_metric="psnr")
    # plot_frame_compression_performance(baseline, quality_metric="ssim")
    plot_superresolution_table(baseline, "database2.json")
    # plot_output_files_size(baseline)

