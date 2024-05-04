import json
import os
import subprocess
import py7zr
import torch

from datasets import UVGDataset
from evaluation.ffmpeg import run_compression, get_bpp_from_ffmpeg, get_psnr_ssim_video_input, get_psnr_ssim_tensor_input
from utils import interpolate_video


def get_keywords(unzip=False):
    return ("Beauty_", "Bosphorus_", "HoneyBee_", "Jockey_", "ReadySetGo_" if unzip else "ReadySteadyGo_",
            "ShakeNDry_", "YachtRide_")


def relevant(filename, unzip=False):
    keywords = get_keywords(unzip=unzip)
    return filename.startswith(keywords) and "1920x1080" in filename


def unzip_uvg(src_path: str):
    """
    This function search for UVG .7z files and unzip them to YUV files. YUV files are saved in the same location where
    they were foundNote that YUV files are 2x more heavy in memory that .7z.
    :param src_path: Path where to find UVG files
    :return: None
    """
    files = list(filter(lambda x: relevant(x, unzip=True) and x.endswith(".7z"), os.listdir(src_path)))
    for file in files:
        print(f"Processing {file}...")
        filepath = os.path.join(src_path, file)
        archive = py7zr.SevenZipFile(filepath, mode='r')
        archive.extractall(src_path)
        archive.close()


def prepare_uvg(src_path: str, dst_path: str):
    """
    Function, which search for UVG YUV files and transform them into iframes. Note that if you just downloaded videos
    from UVG website, you need to unzip them first to use this function
    :param src_path: root path where to search for UVG YUV videos
    :param dst_path: root path where to save UVG videos
    :return: None
    """
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    files = list(filter(lambda x: relevant(x) and x.endswith(".yuv"), os.listdir(src_path)))
    for file in files:
        name = file.split("_")[0]
        savepath = os.path.join(dst_path, name)
        filepath = os.path.join(src_path, file)
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        print(f"Processing {file}...")
        command = ["ffmpeg",
                   "-y",
                   "-pix_fmt", "yuv420p",
                   "-s", "1920x1080",
                   "-i", filepath,
                   f"{savepath}/%03d.png"]
        subprocess.run(command)


@torch.no_grad()
def build_uvg_database(src_path: str, dst_path: str, crfs: list, uvg_path: str,
                       keyframe_interval: int = 0, max_frames: int = 100):
    """
        Function, which search for UVG YUV files and creates database txt file with information about psnr, ssim when
        compressed with hevc/avc with different crf (compression ratio factor) and keyframe interval In addition
        this function evaluates bilinear interpolation. Note that this search only for videos with resolution 1920x1080.
        :param scale: with what scale bilinear interpolation should work
        :param uvg_path: path to uvg dataset root
        :param max_frames: maximum number of frames to take into account
        :param keyframe_interval: how often algorithms should save new keyframe (it has impact on compression ratio
        and compression quality)
        :param crfs: list of compression ratio factors, which should be taken into account when building database
        :param src_path: root path where to search for UVG YUV videos
        :param dst_path: root path where to save database
        :return: None
    """
    if not dst_path.endswith(".json"):
        raise Exception("dst_path must be a valid path to json file")
    dataset = UVGDataset(uvg_path, 2, max_frames=max_frames)
    output = {"keyframe_interval": keyframe_interval, "crfs": crfs, "max_frames": max_frames,
              "resolution": "1920x1024", "data": []}
    files = list(filter(lambda x: relevant(x) and x.endswith(".yuv"), os.listdir(src_path)))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    codecs = ["avc", "hevc"]
    all_iters = len(codecs) * len(files) * len(crfs)
    iteration = 1
    for file in files:
        path = os.path.join(src_path, file)
        name = file.split("_")[0]
        lqs, hqs = dataset.getitem_with_name(name)
        lqs = lqs.to(device).unsqueeze(0)
        hqs = hqs.to(device).unsqueeze(0)
        upscaled = interpolate_video(lqs, size=hqs.shape[-2:])
        psnr_bil, ssim_bil = get_psnr_ssim_tensor_input(upscaled, hqs, nframes=7)
        iteration_result = {"name": name, "bilinear_psnr": psnr_bil, "bilinear_ssim": ssim_bil}
        for codec in codecs:
            iteration_result[codec] = {}
            save_root = f"../outputs/{codec}"
            for crf in crfs:
                mkv_path, ffreport_path = run_compression(path, crf, codec=codec, keyframe_interval=keyframe_interval,
                                                          quite=True, save_root=save_root, max_frames=max_frames)
                bpp_values = get_bpp_from_ffmpeg(ffreport_path, (960, 512))
                psnr_values, ssim_values = get_psnr_ssim_video_input(lqs, mkv_path, nframes=7)

                iteration_result[codec]["crf"] = crf
                iteration_result[codec]["bpp"] = bpp_values
                iteration_result[codec]["psnr"] = psnr_values
                iteration_result[codec]["ssim"] = ssim_values
                print(f"[{iteration}/{all_iters}] {iteration_result}")
                iteration += 1
        output["data"].append(iteration_result)
    with open(dst_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    yuv_files = r"C:\Users\CamaroTheBOSS\Downloads"
    uvg_dataset = r"..\..\Datasets\UVG"
    # unzip_uvg(yuv_files)
    # prepare_uvg(yuv_files, uvg_dataset)
    # build_uvg_database(yuv_files, "database.json",
    #                    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39],
    #                    uvg_dataset, keyframe_interval=105, max_frames=100)
