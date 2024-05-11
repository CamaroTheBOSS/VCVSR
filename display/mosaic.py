import glob
import os
import pathlib
from math import ceil
from typing import List, Union, Tuple
from PIL import Image

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

from datasets import UVGDataset
from evaluation.test import eval_compression
from models.vsrvc import load_model


def _read_data(chkpt_paths: List[str], dataset: UVGDataset, index: int, indexes: torch.Tensor):
    lqs, hqs = dataset.__getitem__(index)
    transform = ToTensor()
    vc_data = [torch.index_select(lqs.unsqueeze(0), 1, indexes)]
    vsr_data = [torch.index_select(hqs.unsqueeze(0), 1, indexes)]
    for chkpt in chkpt_paths:
        print(f"Reading data for checkpoint: {chkpt}")
        vc_root_path = os.path.join(str(pathlib.Path(chkpt).parent), "evaluate_compression")
        vsr_root_path = os.path.join(str(pathlib.Path(chkpt).parent), "evaluate_upscale")
        for data, root_path in zip([vc_data, vsr_data], [vc_root_path, vsr_root_path]):
            all_frame_paths = glob.glob(root_path + "/*.png")
            relevant_frame_paths = [all_frame_paths[i] for i in indexes]
            video = []
            for path in relevant_frame_paths:
                video.append(transform(Image.open(path).convert("RGB")))
            data.append(torch.stack(video).unsqueeze(0))

    return vc_data, vsr_data


def _generate_data(chkpt_paths: List[str], dataset: UVGDataset, index: int, indexes: torch.Tensor):
    lqs = None
    hqs = None
    vc_data, vsr_data = [], []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    indexes = indexes.to(device)
    for chkpt in chkpt_paths:
        print(f"Generating data for checkpoint: {chkpt}")
        model = load_model(chkpt)
        model.eval()
        save_root = str(pathlib.Path(chkpt).parent)
        lqs, hqs, reconstructed, upscaled = eval_compression(model, dataset, examples=[index],
                                                             keyframe_format="jpg", save_root=save_root)
        vc_data.append(torch.index_select(reconstructed, 1, indexes))
        vsr_data.append(torch.index_select(upscaled, 1, indexes))
    vc_data = [torch.index_select(lqs, 1, indexes)] + vc_data
    vsr_data = [torch.index_select(hqs, 1, indexes)] + vsr_data
    return vc_data, vsr_data


@torch.no_grad()
def compression_mosaic(chkpt_paths: List[str], mosaic_like_video: Union[str, int], save_root: str = "./",
                       inter_pad: Tuple[int, int] = (0, 0), generate_data=False, ncols: int = 3):
    dataset = UVGDataset("../../Datasets/UVG", 2)
    index = mosaic_like_video
    if isinstance(mosaic_like_video, str):
        index = dataset.get_index_with_name(mosaic_like_video)

    # Generate data
    indexes = torch.linspace(1, dataset.max_frames - 1, ncols).int()
    if generate_data:
        data, _ = _generate_data(chkpt_paths, dataset, index, indexes)
    else:
        data, _ = _read_data(chkpt_paths, dataset, index, indexes)

    # Mosaic
    rows = []
    _, _, _, h, _ = data[0].shape
    for j, tensor in enumerate(data):
        row = []
        for i, frame in enumerate(tensor.squeeze(0)):
            row.append(frame)
            if inter_pad[1] > 0 and i < (len(tensor.squeeze(0)) - 1):
                row.append(torch.ones(3, h, inter_pad[1]))
        rows.append(torch.cat(row, dim=2))
        _, _, w = rows[0].shape
        if inter_pad[0] > 0 and j < (len(data) - 1):
            rows.append(torch.ones(3, inter_pad[0], w))
    output = torch.cat(rows, dim=1)
    output = np.clip((output.detach().permute(1, 2, 0).cpu().numpy()) * 255., 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    indexes = [str(idx + 1) for idx in indexes.cpu().tolist()]
    filename = f"vc_mosaic_{mosaic_like_video}_frames_{'_'.join(indexes)}.png"
    cv2.imwrite(os.path.join(save_root, filename), output)


@torch.no_grad()
def superresolution_mosaic(chkpt_paths: List[str], mosaic_like_video: Union[str, int], save_root: str = "./",
                           inter_pad: Tuple[int, int] = (0, 0), generate_data=False, frame_idx: int = 50,
                           ncols: int = None, box: Tuple[int, int, int, int] = None):
    dataset = UVGDataset("../../Datasets/UVG", 2)
    index = mosaic_like_video
    if isinstance(mosaic_like_video, str):
        index = dataset.get_index_with_name(mosaic_like_video)

    # Generate data
    indexes = torch.tensor(frame_idx).unsqueeze(0)
    if generate_data:
        _, data = _generate_data(chkpt_paths, dataset, index, indexes)
    else:
        _, data = _read_data(chkpt_paths, dataset, index, indexes)

    # Mosaic
    if ncols is None:
        nrows = 1
        ncols = len(data)
    else:
        nrows = ceil(len(data) / ncols)
    curr_index = 0
    done = False
    rows = []
    b, n, c, h, w = data[0].shape
    if box is not None:
        x, y, w, h = box
    else:
        x, y = 0, 0
    for i in range(nrows):
        row = []
        for j in range(ncols):
            frame = data[curr_index].squeeze(0).squeeze(0)
            frame = frame[:, y:min(y+h, frame.shape[1]), x:min(x+w, frame.shape[2])]
            row.append(frame)
            if inter_pad[1] > 0 and j < (len(frame) - 1):
                row.append(torch.ones(3, h, inter_pad[1]))
            curr_index += 1
            if curr_index == len(data):
                done = True
                break
        row = torch.cat(row, dim=2)
        if i > 0 and row.shape[2] < rows[0].shape[2]:
            row = torch.cat([row, torch.ones(3, h, rows[0].shape[2] - row.shape[2])], dim=2)
        rows.append(row)
        _, _, width = rows[0].shape
        if inter_pad[0] > 0 and i < (len(data) - 1):
            rows.append(torch.ones(3, inter_pad[0], width))
        if done:
            break

    output = torch.cat(rows, dim=1)
    output = np.clip((output.detach().permute(1, 2, 0).cpu().numpy()) * 255., 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    indexes = [str(idx + 1) for idx in indexes.cpu().tolist()]
    filename = f"vsr_mosaic_{mosaic_like_video}_frame_{'_'.join(indexes)}.png"
    cv2.imwrite(os.path.join(save_root, filename), output)

    if box is not None:
        x, y, w, h = box
        output = data[0].squeeze(0).squeeze(0)
        output = np.clip((output.detach().permute(1, 2, 0).cpu().numpy()) * 255., 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = cv2.rectangle(output, (x, y), (min(x+w, output.shape[0]), min(y+h, output.shape[1])), (0, 0, 255), 10)
        cv2.imwrite(os.path.join(save_root, "vsr_mosaic_source.png"), output)


def how_many_layers(checkpoints):
    model = load_model(checkpoints[0])
    print(list(model.parameters()))
    print(len(list(model.parameters())))


if __name__ == "__main__":
    checkpoints = ["../outputs/baseline_no_aug128/model_30.pth", "../outputs/baseline_no_aug512/model_30.pth",
                   "../outputs/baseline_no_aug1024/model_30.pth", "../outputs/baseline_no_aug2048/model_30.pth"]
    how_many_layers(checkpoints)
    # superresolution_mosaic(checkpoints, "YachtRide", ncols=None, save_root="../outputs", inter_pad=(5, 2),
    #                        box=(720, 500, 100, 100))
    # compression_mosaic(checkpoints, "YachtRide", ncols=3, save_root="../outputs", inter_pad=(5, 2))
