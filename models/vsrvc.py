import glob
import os
import pickle
import shutil
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms import ToTensor
from torch import nn
from PIL import Image
from models.basic_blocks import ResidualBlocksWithInputConv, ResidualBlockNoBN
from dcn.deform_conv import DeformConvPack as DCNv1
from models.bit_estimators import HyperpriorEntropyCoder
from models.model_utils import make_layer, count_parameters
from models.upscaling_decoder import UpscalingModule
import gzip

from utils import save_frame


@dataclass
class VSRVCOutput:
    reconstructed: torch.Tensor = None
    upscaled: torch.Tensor = None
    loss_vc: dict = None
    loss_vsr: dict = None
    loss_shared: dict = None
    additional_info: dict = None


@dataclass
class Quantized:
    qint: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor


def load_model(chkpt_path: str = None, name: str = "VSRVC", rdr: int = 128, vc: bool = True, vsr: bool = True,
               quant_type: str = "standard"):
    if chkpt_path is None:
        return VSRVCModel(name, rdr, vc, vsr, quant_type)
    saved_model = torch.load(chkpt_path)
    if "rate_distortion_ratio" in saved_model.keys():
        rdr = saved_model["rate_distortion_ratio"]
    if "model_name" in saved_model.keys():
        name = saved_model["model_name"]
    if "vc" in saved_model.keys():
        vc = saved_model["vc"]
    if "vsr" in saved_model.keys():
        vsr = saved_model["vsr"]
    if "quant_type" in saved_model.keys():
        quant_type = saved_model["quant_type"]
    model = VSRVCModel(name, rdr, vc, vsr, quant_type)
    model.load_state_dict(saved_model['model_state_dict'])
    return model


class MotionEstimator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(MotionEstimator, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        # self.scale_offsets = True

    def forward(self, previous_features: torch.Tensor, current_features: torch.Tensor):
        offsets = torch.cat([previous_features, current_features], dim=1)
        offsets = self.lrelu(self.offset_conv1(offsets))
        offsets = self.lrelu(self.offset_conv2(offsets))
        return offsets
    #     offsets = self.scale(offsets, 2)
    #     return offsets
    #
    # def scale(self, offsets: torch.Tensor, scalar: float):
    #     if offsets.max() < scalar and self.scale_offsets:
    #         offsets = offsets * scalar / offsets.max()
    #     return offsets


def qint_to_float(x: torch.Tensor):
    if x.dtype == torch.qint8:
        scale = x.q_scale()
        zero_point = x.q_zero_point()
        return (x.int_repr().float() - zero_point) * scale
    return x


def int_to_float(x: torch.Tensor, scale: float, zero_point: float):
    return (x.float() - zero_point) * scale


def norm_qint_to_float(x: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor):
    return (x.float() + 0.5) * scales + zero_points


class HyperpriorCompressor(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, quant_type: str = "standard"):
        super(HyperpriorCompressor, self).__init__()
        self.data_encoder = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 5, stride=2, padding=2),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
        ])
        self.data_decoder = nn.Sequential(*[
            nn.ConvTranspose2d(3 * mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.ConvTranspose2d(mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.ConvTranspose2d(mid_channels, out_channels, 5, stride=2, padding=2, output_padding=1),
        ])

        self.hyperprior_encoder = nn.Sequential(*[
            nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
        ])
        self.hyperprior_decoder = nn.Sequential(*[
            nn.ConvTranspose2d(mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(mid_channels, mid_channels * 3 // 2, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(mid_channels * 3 // 2, mid_channels * 2, 3, stride=1, padding=1),
        ])
        self.bit_estimator = HyperpriorEntropyCoder(mid_channels, quant_type)
        self.mid_channels = mid_channels
        self.quant_type = quant_type

    def standard_quantization(self, x: torch.Tensor):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5), None, None
        return torch.round(x), None, None

    def qint_quantization(self, x: torch.Tensor):
        x = torch.clamp(x, -128, 127)
        zero_point = (x.max() + x.min()) / 2
        scale = (x.max() - x.min()) / 128
        if self.training:
            noise = torch.nn.init.uniform_(torch.zeros_like(x), zero_point.item() - scale.item(),
                                           zero_point.item() + scale.item())
            return x + noise, scale, zero_point
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
        return quantized_x, scale, zero_point

    def normalized_quantization(self, x: torch.Tensor):
        maxes = x.view(x.size(0), -1).max(dim=1).values
        mines = x.view(x.size(0), -1).min(dim=1).values
        zero_points = ((maxes + mines) / 2).view(-1, 1, 1, 1)
        scales = ((maxes - mines) / 255).view(-1, 1, 1, 1)

        if self.training:
            noise = (torch.rand_like(x) - 0.5) * scales
            return x + noise, scales, zero_points
        return torch.round(((x - zero_points) / scales - 0.5)).to(torch.int8), scales, zero_points

    def quantize(self, x: torch.Tensor):
        if self.quant_type == "standard":
            return self.standard_quantization(x)
        elif self.quant_type == "qint":
            return self.qint_quantization(x)
        elif self.quant_type == "normalized":
            return self.normalized_quantization(x)

    def get_float(self, x: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor):
        if self.training:
            return x
        elif self.quant_type == "standard":
            return x
        elif self.quant_type == "qint":
            return qint_to_float(x)
        elif self.quant_type == "normalized":
            return (x.float() + 0.5) * scales + zero_points

    def train_compression_decompression(self, data: torch.Tensor):
        data_prior = self.data_encoder(data)
        qint_data_prior, scale_prior, zero_point_prior = self.quantize(data_prior)
        quantized_data_prior = self.get_float(qint_data_prior, scale_prior, zero_point_prior)

        data_hyperprior = self.hyperprior_encoder(data_prior)
        qint_data_hyperprior, scale_hyperprior, zero_point_hyperprior = self.quantize(data_hyperprior)
        quantized_data_hyperprior = self.get_float(qint_data_hyperprior, scale_hyperprior, zero_point_hyperprior)

        mu_sigmas = self.hyperprior_decoder(quantized_data_hyperprior)
        prior_bits, hyperprior_bits, _, _ = self.bit_estimator(quantized_data_prior, quantized_data_hyperprior,
                                                               mu_sigmas, scale_prior, zero_point_prior,
                                                               scale_hyperprior, zero_point_hyperprior)

        # sigmas = mu_sigmas[:, quantized_data_prior.shape[1]:]
        reconstructed_data = self.data_decoder(torch.cat([quantized_data_prior, mu_sigmas], dim=1))
        count_nonzeros = torch.count_nonzero(quantized_data_prior) + torch.count_nonzero(quantized_data_hyperprior)
        return reconstructed_data, prior_bits, hyperprior_bits, count_nonzeros

    def compress(self, data: torch.Tensor):
        data_prior = self.data_encoder(data)
        quantized_data_prior, scale_prior, zero_point_prior = self.quantize(data_prior)
        q_prior = Quantized(quantized_data_prior, scale_prior, zero_point_prior)

        data_hyperprior = self.hyperprior_encoder(data_prior)
        quantized_data_hyperprior, scale_hyperprior, zero_point_hyperprior = self.quantize(data_hyperprior)
        q_hyperprior = Quantized(quantized_data_hyperprior, scale_hyperprior, zero_point_hyperprior)

        return q_prior, q_hyperprior

    def decompress(self, prior: Quantized, hyperprior: Quantized, return_mu_sigmas=False):
        quantized_data_prior = self.get_float(prior.qint, prior.scales, prior.zero_points)
        quantized_data_hyperprior = self.get_float(hyperprior.qint, hyperprior.scales, hyperprior.zero_points)
        mu_sigmas = self.hyperprior_decoder(quantized_data_hyperprior)
        reconstructed_data = self.data_decoder(torch.cat([quantized_data_prior, mu_sigmas], dim=1))
        if return_mu_sigmas:
            return reconstructed_data, mu_sigmas
        return reconstructed_data


class MotionCompensator(nn.Module):
    def __init__(self, channels: int, dcn_groups: int = 8):
        super(MotionCompensator, self).__init__()
        self.deformable_convolution = DCNv1(channels, channels, 3, stride=1, padding=1, dilation=1,
                                            deformable_groups=dcn_groups, extra_offset_mask=True)
        self.refine_conv1 = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.refine_conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, previous_features: torch.Tensor, offsets: torch.Tensor):
        aligned_features = self.deformable_convolution([previous_features, offsets])

        refined_features = torch.cat([aligned_features, previous_features], dim=1)
        refined_features = self.lrelu(self.refine_conv1(refined_features))
        refined_features = self.lrelu(self.refine_conv2(refined_features))
        return aligned_features + refined_features


class CompressionReconstructionHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super(CompressionReconstructionHead, self).__init__()
        self.reconstruction_trunk = ResidualBlocksWithInputConv(in_channels, mid_channels, num_blocks=3)
        self.deconv = nn.ConvTranspose2d(mid_channels, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, features: torch.Tensor):
        reconstruction = self.reconstruction_trunk(features)
        reconstruction = self.deconv(reconstruction)
        return reconstruction


class VSRVCModel(nn.Module):
    def __init__(self, name="", rdr: float = 128, vc: bool = True, vsr: bool = True, quant_type: str = "standard"):
        super(VSRVCModel, self).__init__()
        self.feature_extraction = nn.Sequential(*[
            nn.Conv2d(3, 64, 5, 2, 2),
            ResidualBlocksWithInputConv(in_channels=64, out_channels=64, num_blocks=3)
        ])
        self.motion_estimator = MotionEstimator(in_channels=64, out_channels=64)
        self.motion_compressor = HyperpriorCompressor(in_channels=64, mid_channels=128, out_channels=64,
                                                      quant_type=quant_type)
        self.motion_compensator = MotionCompensator(channels=64, dcn_groups=8)
        self.residual_compressor = HyperpriorCompressor(in_channels=64, mid_channels=128, out_channels=64,
                                                        quant_type=quant_type)

        self.reconstruction_head = CompressionReconstructionHead(in_channels=64, mid_channels=64)
        self.upscaler_head = UpscalingModule(in_channels=64, mid_channels=64)

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
              f"    Motion estimator   : {count_parameters(self.motion_estimator)}\n" \
              f"    Motion compressor  : {count_parameters(self.motion_compressor)}\n" \
              f"    Motion compensator : {count_parameters(self.motion_compensator)}\n" \
              f"    Residual compressor: {count_parameters(self.residual_compressor)}\n" \
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
        to_code = {"offsets_prior": [], "offsets_hyperprior": [], "residuals_prior": [], "residuals_hyperprior": []}
        if self.quant_type in ["qint", "normalized"]:
            to_code["quantization_info"] = []

        start_time = time.time()
        if verbose:
            print(f"Starting inference...")
        current_features = self.feature_extraction(video[:, 0])
        previous_features = current_features
        offsets = self.motion_estimator(previous_features, current_features)
        offsets_prior, offsets_hyperprior = self.motion_compressor.compress(offsets)
        decompressed_offsets = self.motion_compressor.decompress(offsets_prior, offsets_hyperprior)
        aligned_features = self.motion_compensator(previous_features, decompressed_offsets)
        hqs = [self.upscaler_head(aligned_features, video[:, 0])]
        for i in range(1, video.shape[1]):
            current_lqf = video[:, i]
            current_features = self.feature_extraction(current_lqf)
            offsets = self.motion_estimator(previous_features, current_features)
            offsets_prior, offsets_hyperprior = self.motion_compressor.compress(offsets)
            decompressed_offsets = self.motion_compressor.decompress(offsets_prior, offsets_hyperprior)
            aligned_features = self.motion_compensator(previous_features, decompressed_offsets)
            residuals = current_features - aligned_features
            residual_prior, residual_hyperprior = self.residual_compressor.compress(residuals)
            hqs.append(self.upscaler_head(aligned_features, current_lqf))
            if self.quant_type in ["qint", "normalized"]:
                to_code["quantization_info"].append(
                    torch.tensor([[offsets_prior.scales, offsets_prior.zero_points],
                                  [offsets_hyperprior.scales, offsets_hyperprior.zero_points],
                                  [residual_prior.scales, residual_prior.zero_points],
                                  [residual_hyperprior.scales, residual_hyperprior.zero_points]])
                )
                offsets_prior.qint = offsets_prior.qint.int_repr() if self.quant_type == "qint" else offsets_prior.qint
                offsets_hyperprior.qint = offsets_hyperprior.qint.int_repr() if self.quant_type == "qint" else offsets_hyperprior.qint
                residual_prior.qint = residual_prior.qint.int_repr() if self.quant_type == "qint" else residual_prior.qint
                residual_hyperprior.qint = residual_hyperprior.qint.int_repr() if self.quant_type == "qint" else residual_hyperprior.qint

            to_code["offsets_prior"].append(offsets_prior.qint)
            to_code["offsets_hyperprior"].append(offsets_hyperprior.qint)
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
        pkl_filenames = ["offsets_prior", "offsets_hyperprior", "residuals_prior", "residuals_hyperprior"]
        if self.quant_type in ["qint", "normalized"]:
            pkl_filenames.append("quantization_info")
        listed_files = os.listdir(compressed_root)
        pkl_files = list(filter(lambda file: any(pkl in file for pkl in pkl_filenames), listed_files))
        keyframe = list(filter(lambda x: x.startswith("keyframe"), listed_files))
        if len(pkl_files) != (5 if self.quant_type in ["qint", "normalized"] else 4):
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
                decoded_data["offsets_prior"].size(0))]  # for compatibility purposes
        for op, ohp, rp, rhp, q_info in zip(decoded_data["offsets_prior"], decoded_data["offsets_hyperprior"],
                                            decoded_data["residuals_prior"], decoded_data["residuals_hyperprior"],
                                            decoded_data["quantization_info"]):
            op = Quantized(op.unsqueeze(0), *q_info[0])
            ohp = Quantized(ohp.unsqueeze(0), *q_info[1])
            rp = Quantized(rp.unsqueeze(0), *q_info[2])
            rhp = Quantized(rhp.unsqueeze(0), *q_info[3])
            reconstructed_offsets = self.motion_compressor.decompress(op, ohp)
            aligned_features = self.motion_compensator(previous_features, reconstructed_offsets)

            reconstructed_residuals = self.residual_compressor.decompress(rp, rhp)
            reconstructed_features = reconstructed_residuals + aligned_features

            reconstructed_frame = self.reconstruction_head(reconstructed_features)
            decoded_frames.append(reconstructed_frame)
            previous_features = self.feature_extraction(reconstructed_frame)

        if return_data:
            return torch.stack(decoded_frames).squeeze(1).unsqueeze(0), decoded_data
        return torch.stack(decoded_frames).squeeze(1).unsqueeze(0)

    def generate_video(self, previous_frame: torch.Tensor, current_frame: torch.Tensor, nframes: int = 6):
        decoded_frames = [previous_frame, current_frame]
        previous_features = self.feature_extraction(previous_frame)
        current_features = self.feature_extraction(current_frame)
        offsets = self.motion_estimator(previous_features, current_features)
        offsets_prior, offsets_hyperprior = self.motion_compressor.compress(offsets)
        decompressed_offsets, mu_sigmas_offsets = (
            self.motion_compressor.decompress(offsets_prior, offsets_hyperprior, return_mu_sigmas=True)
        )
        aligned_features = self.motion_compensator(previous_features, decompressed_offsets)
        residuals = current_features - aligned_features
        residual_prior, residual_hyperprior = self.residual_compressor.compress(residuals)
        decompressed_residuals, mu_sigmas_residuals = (
            self.residual_compressor.decompress(residual_prior, residual_hyperprior, return_mu_sigmas=True)
        )
        for i in range(nframes):
            mu_offsets, sigma_offsets = self.motion_compressor.bit_estimator.get_mu_sigma(mu_sigmas_offsets)
            mu_residuals, sigma_residuals = self.residual_compressor.bit_estimator.get_mu_sigma(mu_sigmas_residuals)
            offsets_prior.qint = torch.normal(mu_offsets, sigma_offsets)
            offsets_hyperprior.qint = self.motion_compressor.bit_estimator.sample_hyperprior(
                n_values=offsets_hyperprior.qint.shape[-2] * offsets_hyperprior.qint.shape[-1], low=-25, high=25,
                precision=3000
            ).reshape(offsets_hyperprior.qint.shape)
            residuals_prior = torch.normal(mu_residuals, sigma_residuals)
            residuals_hyperprior = self.residual_compressor.bit_estimator.sample_hyperprior(
                n_values=offsets_hyperprior.qint.shape[-2] * offsets_hyperprior.qint.shape[-1], low=-25, high=25,
                precision=3000
            ).reshape(offsets_hyperprior.qint.shape)
            residuals_prior = Quantized(residuals_prior, scales=offsets_prior.scales,
                                        zero_points=offsets_prior.zero_points)
            residuals_hyperprior = Quantized(residuals_hyperprior, scales=offsets_hyperprior.scales,
                                             zero_points=offsets_hyperprior.zero_points)
            reconstructed_offsets = self.motion_compressor.decompress(offsets_prior, offsets_hyperprior)
            aligned_features = self.motion_compensator(current_features, reconstructed_offsets)

            reconstructed_residuals = self.residual_compressor.decompress(residuals_prior, residuals_hyperprior)
            reconstructed_features = reconstructed_residuals + aligned_features

            reconstructed_frame = self.reconstruction_head(reconstructed_features)
            decoded_frames.append(reconstructed_frame)
            current_features = self.feature_extraction(reconstructed_frame)

        return torch.stack(decoded_frames).squeeze(1).unsqueeze(0)

    def generate_video2(self, previous_frame: torch.Tensor, current_frame: torch.Tensor, nframes: int = 6):
        decoded_frames = [previous_frame, current_frame]
        previous_features = self.feature_extraction(previous_frame)
        for i in range(nframes):
            current_features = self.feature_extraction(current_frame)
            offsets = self.motion_estimator(previous_features, current_features)
            offsets_prior, offsets_hyperprior = self.motion_compressor.compress(offsets)
            decompressed_offsets, mu_sigmas_offsets = (
                self.motion_compressor.decompress(offsets_prior, offsets_hyperprior, return_mu_sigmas=True)
            )
            aligned_features = self.motion_compensator(previous_features, decompressed_offsets)
            residuals = current_features - aligned_features
            residual_prior, residual_hyperprior = self.residual_compressor.compress(residuals)
            decompressed_residuals, mu_sigmas_residuals = (
                self.residual_compressor.decompress(residual_prior, residual_hyperprior, return_mu_sigmas=True)
            )
            mu_offsets, sigma_offsets = self.motion_compressor.bit_estimator.get_mu_sigma(mu_sigmas_offsets)
            mu_residuals, sigma_residuals = self.residual_compressor.bit_estimator.get_mu_sigma(mu_sigmas_residuals)
            offsets_prior = torch.normal(mu_offsets, sigma_offsets)
            offsets_hyperprior = self.motion_compressor.bit_estimator.sample_hyperprior(
                n_values=offsets_hyperprior.shape[-2] * offsets_hyperprior.shape[-1], low=-25, high=25, precision=3000
            ).reshape(offsets_hyperprior.shape)
            residuals_prior = torch.normal(mu_residuals, sigma_residuals)
            residuals_hyperprior = self.residual_compressor.bit_estimator.sample_hyperprior(
                n_values=offsets_hyperprior.shape[-2] * offsets_hyperprior.shape[-1], low=-25, high=25, precision=3000
            ).reshape(offsets_hyperprior.shape)

            reconstructed_offsets = self.motion_compressor.decompress(offsets_prior, offsets_hyperprior)
            aligned_features = self.motion_compensator(current_features, reconstructed_offsets)

            reconstructed_residuals = self.residual_compressor.decompress(residuals_prior, residuals_hyperprior)
            reconstructed_features = reconstructed_residuals + aligned_features

            reconstructed_frame = self.reconstruction_head(reconstructed_features)
            decoded_frames.append(reconstructed_frame)
            previous_features = current_features
            current_frame = reconstructed_frame

        return torch.stack(decoded_frames).squeeze(1).unsqueeze(0)

    def forward(self, previous_lqf: torch.Tensor, current_lqf: torch.Tensor, current_hqf: torch.Tensor = None):
        # [SHARED] Get into feature space
        previous_features = self.feature_extraction(previous_lqf)
        current_features = self.feature_extraction(current_lqf)

        # [SHARED] Estimate motion, compress motion, decompress motion and align previous features
        offsets = self.motion_estimator(previous_features, current_features)
        decompressed_offsets, bits_offsets_prior, bits_offsets_hyperprior, count_nonzeros_offsets = (
            self.motion_compressor.train_compression_decompression(offsets)
        )
        aligned_features = self.motion_compensator(previous_features, decompressed_offsets)
        aligned_frame = self.reconstruction_head(aligned_features)
        current_frame = self.reconstruction_head(current_features)
        loss_shared = {
            "shared_recon_alignment": self.rdr * self.reconstruction_loss(aligned_frame, current_lqf),
            "shared_recon_features": self.rdr * self.reconstruction_loss(current_frame, current_lqf),
        }

        # [VC branch] Get residual, compress it and decompress
        loss_vc = {}
        reconstructed_frame = torch.zeros_like(current_lqf)
        count_nonzeros_residuals = 0
        if self.vc:
            residuals = current_features - aligned_features
            decompressed_residuals, bits_residuals_prior, bits_residuals_hyperprior, count_nonzeros_residuals = (
                self.residual_compressor.train_compression_decompression(residuals)
            )

            # [VC branch] Reconstruct features from residual and reconstruct frame
            reconstructed_features = decompressed_residuals + aligned_features
            reconstructed_frame = self.reconstruction_head(reconstructed_features)

            # [VC branch] Loss function
            frame_area = current_lqf.shape[0] * current_lqf.shape[2] * current_lqf.shape[3]  # B * H * W
            loss_vc = {
                "vc_recon_compression": self.rdr * self.reconstruction_loss(reconstructed_frame, current_lqf),
                "vc_bits_offsets_prior": bits_offsets_prior / frame_area,
                "vc_bits_offsets_hyperprior": bits_offsets_hyperprior / frame_area,
                "vc_bits_residuals_prior": bits_residuals_prior / frame_area,
                "vc_bits_residuals_hyperprior": bits_residuals_hyperprior / frame_area,
            }

        # [SR branch] Super-resolution upscale
        loss_vsr = {}
        upscaled_frame = torch.zeros_like(current_lqf)
        if self.vsr:
            upscaled_frame = self.upscaler_head(aligned_features, current_lqf)

            # [SR branch] Loss function
            loss_vsr = {
                "vsr_recon": self.reconstruction_loss(upscaled_frame,
                                                      current_hqf) if current_hqf is not None else None,
            }

        additional_info = {"count_non_zeros_offsets": count_nonzeros_offsets,
                           "count_non_zeros_residuals": count_nonzeros_residuals}
        return VSRVCOutput(reconstructed_frame, upscaled_frame, loss_vc, loss_vsr, loss_shared, additional_info)
