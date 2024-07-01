import glob
import gzip
import os
import pickle
import shutil
import time

import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from PIL import Image
from models.basic_blocks import ResidualBlocksWithInputConv
from models.model_utils import count_parameters, Quantized, VSRVCOutput
from models.upscaling_decoder import UpscalingModule
from models.vsrvc import HyperpriorCompressor, CompressionReconstructionHead
from utils import save_frame


class VSRVCNoAlignment(nn.Module):
    def __init__(self, name="", rdr: float = 128, vc: bool = True, vsr: bool = True, quant_type: str = "standard"):
        super(VSRVCNoAlignment, self).__init__()
        self.feature_extraction = nn.Sequential(*[
            nn.Conv2d(3, 64, 5, 2, 2),
            ResidualBlocksWithInputConv(in_channels=64, out_channels=64, num_blocks=3)
        ])
        self.residual_compressor = HyperpriorCompressor(in_channels=64, mid_channels=128, out_channels=64,
                                                        quant_type=quant_type)

        self.reconstruction_head = CompressionReconstructionHead(in_channels=64, mid_channels=64)
        self.upscaler_head = UpscalingModule(in_channels=128, mid_channels=64)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

        self.reconstruction_loss = torch.nn.L1Loss()  # MSELoss
        self.rdr = rdr
        self.name = name
        self.vc = vc
        self.vsr = vsr
        self.quant_type = quant_type

    def save_chkpt(self, root: str, epoch: int, only_last: bool = True):
        kwargs = {"model_name": self.name, "rate_distortion_ratio": self.rdr, "vc": self.vc, "vsr": self.vsr,
                  "quant_type": self.quant_type}
        if only_last:
            pth_files = glob.glob(os.path.join(root, "model_*.pth"))
            for file in pth_files:
                os.remove(file)
        path = os.path.join(root, f"model_{epoch + 1}.pth")
        torch.save({'model_state_dict': self.state_dict(), **kwargs}, path)

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def summary(self):
        txt = f"Attributes:\n" \
              f"    name: {self.name}\n" \
              f"    rate distortion ratio: {self.rdr}\n" \
              f"    VC: {self.vc}\n" \
              f"    VSR: {self.vsr}\n" \
              f"    reconstruction loss: {self.reconstruction_loss}\n" \
              "Parameters:\n" \
              f"    Feature extractor  : {count_parameters(self.feature_extraction)}\n" \
              f"    Residual compressor  : {count_parameters(self.residual_compressor)}\n" \
              f"    Reconstruction head: {count_parameters(self.reconstruction_head)}\n" \
              f"    Upscaler head      : {count_parameters(self.upscaler_head)}\n" \
              f"    TOTAL              : {count_parameters(self)}\n"
        print(txt)
        return txt

    def compress(self, video: torch.Tensor, save_root: str = "./", keyframe_format="jpg", verbose=False,
                 aggregate_bpp="mean", folder_name: str = "compressed_data", return_data=False):
        if save_root is None:
            save_root = "./"
        b, n, c, h, w = video.shape
        video = video.to(self.device)
        to_code = {"residuals_prior": [], "residuals_hyperprior": []}
        if self.quant_type in ["qint", "normalized"]:
            to_code["quantization_info"] = []

        start_time = time.time()
        if verbose:
            print(f"Starting inference...")
        current_features = self.feature_extraction(video[:, 0])
        previous_features = current_features
        merged_features = torch.cat([previous_features, current_features], dim=1)
        hqs = [self.upscaler_head(merged_features, video[:, 0])]
        for i in range(1, video.shape[1]):
            current_lqf = video[:, i]
            current_features = self.feature_extraction(current_lqf)
            residuals = current_features - previous_features
            residual_prior, residual_hyperprior = self.residual_compressor.compress(residuals)
            merged_features = torch.cat([previous_features, current_features], dim=1)
            hqs.append(self.upscaler_head(merged_features, current_lqf))
            if self.quant_type in ["qint", "normalized"]:
                to_code["quantization_info"].append(
                    torch.tensor([[residual_prior.scales, residual_prior.zero_points],
                                  [residual_hyperprior.scales, residual_hyperprior.zero_points]])
                )
                residual_prior.qint = residual_prior.qint.int_repr() if self.quant_type == "qint" else residual_prior.qint
                residual_hyperprior.qint = residual_hyperprior.qint.int_repr() if self.quant_type == "qint" else residual_hyperprior.qint

            to_code["residuals_prior"].append(residual_prior.qint)
            to_code["residuals_hyperprior"].append(residual_hyperprior.qint)
            previous_features = current_features
        if verbose:
            print(f"Model inference done")

        output_dir = os.path.join(save_root, folder_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        size_bpp = []
        if verbose:
            print(f"Compressing tensors...")
        items = to_code.items()
        for i, (key, value) in enumerate(items):
            filepath = os.path.join(output_dir, f"{key}.pkl")
            data = torch.stack(value).squeeze(1)
            if key != "quantization_info":
                data = data.int()
                if torch.max(data) <= 127 and torch.min(data) >= -128:
                    data = data.to(torch.int8)
            with gzip.open(filepath, "wb") as f:
                pickle.dump(data, f)
            if verbose:
                print(f"{round((i + 1) / len(items) * 100, 2)}%")
            size_bpp.append(os.stat(filepath).st_size * 8. / (n * h * w))
        if keyframe_format != "none":
            if verbose:
                print(f"Compressing keyframe...")
            save_frame(os.path.join(output_dir, f"keyframe.{keyframe_format}"), video[0, 0])
            size_bpp.insert(0,
                            os.stat(os.path.join(output_dir, f"keyframe.{keyframe_format}")).st_size * 8. / (n * h * w))
        if verbose:
            print(f"Done in {time.time() - start_time} seconds")
        if aggregate_bpp == "mean":
            size_bpp = np.array(size_bpp).sum()

        if return_data:
            return output_dir, size_bpp, torch.stack(hqs).squeeze(1).unsqueeze(0), to_code
        return output_dir, size_bpp, torch.stack(hqs).squeeze(1).unsqueeze(0)

    def decompress(self, compressed_root: str, return_data: bool = False):
        pkl_filenames = ["residuals_prior", "residuals_hyperprior"]
        if self.quant_type in ["qint", "normalized"]:
            pkl_filenames.append("quantization_info")
        listed_files = os.listdir(compressed_root)
        pkl_files = list(filter(lambda file: any(pkl in file for pkl in pkl_filenames), listed_files))
        keyframe = list(filter(lambda x: x.startswith("keyframe"), listed_files))
        if len(pkl_files) != (3 if self.quant_type in ["qint", "normalized"] else 2):
            raise Exception("Error! Lack of necessary pkl files needed for decompression.")
        if len(keyframe) != 1:
            raise Exception("Error! Lack of necessary keyframe image file needed for decompression")
        decoded_data = {}
        for filename in pkl_files:
            with gzip.open(os.path.join(compressed_root, filename), "rb") as f:
                decoded_data[filename[:-4]] = pickle.load(f).to(self.device).float()
        decoded_frames = [ToTensor()(Image.open(os.path.join(compressed_root, keyframe[0])).
                                     convert("RGB")).to(self.device).unsqueeze(0)]
        previous_features = self.feature_extraction(decoded_frames[0])

        if self.quant_type == "standard":
            decoded_data["quantization_info"] = [[[None, None] for i in range(4)] for j in range(
                decoded_data["residuals_prior"].size(0))]  # for compatibility purposes
        for rp, rhp, q_info in zip(decoded_data["residuals_prior"], decoded_data["residuals_hyperprior"],
                                   decoded_data["quantization_info"]):
            rp = Quantized(rp.unsqueeze(0), *q_info[2])
            rhp = Quantized(rhp.unsqueeze(0), *q_info[3])

            reconstructed_residuals = self.residual_compressor.decompress(rp, rhp)
            reconstructed_features = reconstructed_residuals + previous_features

            reconstructed_frame = self.reconstruction_head(reconstructed_features)
            decoded_frames.append(reconstructed_frame)
            previous_features = self.feature_extraction(reconstructed_frame)

        if return_data:
            return torch.stack(decoded_frames).squeeze(1).unsqueeze(0), decoded_data
        return torch.stack(decoded_frames).squeeze(1).unsqueeze(0)

    def forward(self, previous_lqf: torch.Tensor, current_lqf: torch.Tensor, current_hqf: torch.Tensor = None):
        # [SHARED] Get into feature space
        previous_features = self.feature_extraction(previous_lqf)
        current_features = self.feature_extraction(current_lqf)

        # [SHARED] Estimate motion, compress motion, decompress motion and align previous features
        previous_frame = self.reconstruction_head(previous_features)
        current_frame = self.reconstruction_head(current_features)
        loss_shared = {
            "shared_recon_alignment": torch.tensor(0),
            "shared_recon_prev": self.rdr * self.reconstruction_loss(previous_frame, previous_lqf),
            "shared_recon_features": self.rdr * self.reconstruction_loss(current_frame, current_lqf),
        }

        # [VC branch] Get residual, compress it and decompress
        loss_vc = {}
        reconstructed_frame = torch.zeros_like(current_lqf)
        count_nonzeros_residuals = 0
        if self.vc:
            residuals = current_features - previous_features
            decompressed_residuals, bits_residuals_prior, bits_residuals_hyperprior, count_nonzeros_residuals = (
                self.residual_compressor.train_compression_decompression(residuals)
            )

            # [VC branch] Reconstruct features from residual and reconstruct frame
            reconstructed_features = decompressed_residuals + previous_features
            reconstructed_frame = self.reconstruction_head(reconstructed_features)

            # [VC branch] Loss function
            frame_area = current_lqf.shape[0] * current_lqf.shape[2] * current_lqf.shape[3]  # B * H * W
            loss_vc = {
                "vc_recon_compression": self.rdr * self.reconstruction_loss(reconstructed_frame, current_lqf),
                "vc_bits_offsets_prior": torch.tensor(0),
                "vc_bits_offsets_hyperprior": torch.tensor(0),
                "vc_bits_residuals_prior": bits_residuals_prior / frame_area,
                "vc_bits_residuals_hyperprior": bits_residuals_hyperprior / frame_area,
            }

        # [SR branch] Super-resolution upscale
        loss_vsr = {}
        upscaled_frame = torch.zeros_like(current_lqf)
        if self.vsr:
            merged_features = torch.cat([previous_features, current_features], dim=1)
            upscaled_frame = self.upscaler_head(merged_features, current_lqf)

            # [SR branch] Loss function
            loss_vsr = {
                "vsr_recon": self.reconstruction_loss(upscaled_frame,
                                                      current_hqf) if current_hqf is not None else None,
            }

        additional_info = {"count_non_zeros_offsets": 0, "count_non_zeros_residuals": count_nonzeros_residuals}
        return VSRVCOutput(reconstructed_frame, upscaled_frame, loss_vc, loss_vsr, loss_shared, additional_info)