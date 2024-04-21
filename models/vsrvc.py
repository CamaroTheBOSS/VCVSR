import glob
import os
import pickle
import shutil
from dataclasses import dataclass

import torch
from torchvision.transforms import ToTensor
from torch import nn
from PIL import Image
from models.basic_blocks import ResidualBlocksWithInputConv, ResidualBlockNoBN
from dcn.deform_conv import DeformConvPack as DCNv1
from models.bit_estimators import HyperpriorEntropyCoder
from models.model_utils import make_layer, count_parameters
from models.upscaling_decoder import UpscalingModule
import gzip
from kornia.augmentation import Resize

from utils import save_frame


@dataclass
class VSRVCOutput:
    reconstructed: torch.Tensor = None
    upscaled: torch.Tensor = None
    loss_vc: dict = None
    loss_vsr: dict = None


def load_model(chkpt_path: str = None, rate_distortion_ratio: int = 128):
    model = VSRVCModel(rate_distortion_ratio)
    if chkpt_path is not None:
        saved_model = torch.load(chkpt_path)
        model.load_state_dict(saved_model['model_state_dict'])
    return model


class MotionEstimator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(MotionEstimator, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, previous_features: torch.Tensor, current_features: torch.Tensor):
        offsets = torch.cat([previous_features, current_features], dim=1)
        offsets = self.lrelu(self.offset_conv1(offsets))
        offsets = self.lrelu(self.offset_conv2(offsets))
        return offsets


class HyperpriorCompressor(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(HyperpriorCompressor, self).__init__()
        self.data_encoder = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 5, stride=2, padding=2),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
        ])
        self.data_decoder = nn.Sequential(*[
            nn.ConvTranspose2d(mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
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
        self.bit_estimator = HyperpriorEntropyCoder(mid_channels)
        self.mid_channels = mid_channels

        # self.init_weights()

    # def init_weights(self):
    #     torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (2 + 64) / (4))))
    #     torch.nn.init.constant_(self.conv1.bias.data, 0.01)

    def quantize(self, x: torch.Tensor):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        return torch.round(x)

    def train_compression_decompression(self, data: torch.Tensor):
        data_prior = self.data_encoder(data)
        quantized_data_prior = self.quantize(data_prior)

        data_hyperprior = self.hyperprior_encoder(data_prior)
        quantized_data_hyperprior = self.quantize(data_hyperprior)

        mu_sigmas = self.hyperprior_decoder(quantized_data_hyperprior)
        prior_bits, hyperprior_bits = self.bit_estimator(quantized_data_prior, quantized_data_hyperprior, mu_sigmas)

        reconstructed_data = self.data_decoder(quantized_data_prior)
        return reconstructed_data, prior_bits, hyperprior_bits

    def compress(self, data: torch.Tensor):
        data_prior = self.data_encoder(data)
        quantized_data_prior = self.quantize(data_prior)

        data_hyperprior = self.hyperprior_encoder(data_prior)
        quantized_data_hyperprior = self.quantize(data_hyperprior)

        # mu_sigmas = self.hyperprior_decoder(quantized_data_hyperprior) ???
        return quantized_data_prior, quantized_data_hyperprior

    def decompress(self, quantized_data_prior: torch.Tensor, quantized_data_hyperprior: torch.Tensor):
        # mu_sigmas = self.hyperprior_decoder(quantized_data_hyperprior) ???
        reconstructed_data = self.data_decoder(quantized_data_prior)
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
    def __init__(self, rate_distortion_ratio: float = 128):
        super(VSRVCModel, self).__init__()
        self.feature_extraction = nn.Sequential(*[
            nn.Conv2d(3, 64, 5, 2, 2),
            ResidualBlocksWithInputConv(in_channels=64, out_channels=64, num_blocks=3)
        ])
        self.motion_estimator = MotionEstimator(in_channels=64, out_channels=64)
        self.motion_compressor = HyperpriorCompressor(in_channels=64, mid_channels=128, out_channels=64)
        self.motion_compensator = MotionCompensator(channels=64, dcn_groups=8)
        self.residual_compressor = HyperpriorCompressor(in_channels=64, mid_channels=128, out_channels=64)

        self.reconstruction_head = CompressionReconstructionHead(in_channels=64, mid_channels=64)
        self.upscaler_head = UpscalingModule(in_channels=64, mid_channels=64)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

        self.reconstruction_loss = torch.nn.L1Loss()  # MSELoss
        self.rdr = rate_distortion_ratio

    def summary(self):
        print("Parameters:"
              f"    Feature extractor  : {count_parameters(self.feature_extraction)}\n"
              f"    Motion estimator   : {count_parameters(self.motion_estimator)}\n"
              f"    Motion compressor  : {count_parameters(self.motion_compressor)}\n"
              f"    Motion compensator : {count_parameters(self.motion_compensator)}\n"
              f"    Residual compressor: {count_parameters(self.residual_compressor)}\n"
              f"    Reconstruction head: {count_parameters(self.reconstruction_head)}\n"
              f"    Upscaler head      : {count_parameters(self.upscaler_head)}\n"
              f"    TOTAL              : {count_parameters(self)}\n"
              )

    def compress(self, video: torch.Tensor, save_root: str = "./", keyframe_format="jpg"):
        video = video.to(self.device)
        to_code = {"offsets_prior": [], "offsets_hyperprior": [], "residuals_prior": [], "residuals_hyperprior": []}
        previous_features = self.feature_extraction(video[:, 0])
        b, n, c, h, w = video.shape
        hqs = []
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

            to_code["offsets_prior"].append(offsets_prior)
            to_code["offsets_hyperprior"].append(offsets_hyperprior)
            to_code["residuals_prior"].append(residual_prior)
            to_code["residuals_hyperprior"].append(residual_hyperprior)
            previous_features = current_features

        output_dir = os.path.join(save_root, "compressed_data")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        size_bytes = 0
        for key, value in to_code.items():
            filepath = os.path.join(output_dir, f"{key}.pkl")
            with gzip.open(filepath, "wb") as f:
                data = torch.stack(value).squeeze(1)
                pickle.dump(data, f)
            size_bytes += os.stat(filepath).st_size * 8.
        save_frame(os.path.join(output_dir, f"keyframe.{keyframe_format}"),  video[0, 0])
        size_bytes += os.stat(os.path.join(output_dir, f"keyframe.{keyframe_format}")).st_size * 8.

        return output_dir, size_bytes / (n * h * w), torch.stack(hqs).squeeze(1).unsqueeze(0)

    def decompress(self, compressed_root: str):
        pkl_filenames = ["offsets_prior", "offsets_hyperprior", "residuals_prior", "residuals_hyperprior"]
        listed_files = os.listdir(compressed_root)
        pkl_files = list(filter(lambda file: any(pkl in file for pkl in pkl_filenames), listed_files))
        keyframe = list(filter(lambda x: x.startswith("keyframe"), listed_files))
        if len(pkl_files) != 4:
            raise Exception("Error! Lack of necessary pkl files needed for decompression.")
        if len(keyframe) != 1:
            raise Exception("Error! Lack of necessary keyframe image file needed for decompression")
        decoded_data = {}
        for filename in pkl_files:
            with gzip.open(os.path.join(compressed_root, filename), "rb") as f:
                decoded_data[filename[:-4]] = pickle.load(f).to(self.device)
        decoded_frames = [ToTensor()(Image.open(os.path.join(compressed_root, keyframe[0])).
                                     convert("RGB")).to(self.device).unsqueeze(0)]
        previous_features = self.feature_extraction(decoded_frames[0])

        for op, ohp, rp, rhp in zip(decoded_data["offsets_prior"], decoded_data["offsets_hyperprior"],
                                    decoded_data["residuals_prior"], decoded_data["residuals_hyperprior"]):
            op, ohp, rp, rhp = op.unsqueeze(0), ohp.unsqueeze(0), rp.unsqueeze(0), rhp.unsqueeze(0)
            reconstructed_offsets = self.motion_compressor.decompress(op, ohp)
            aligned_features = self.motion_compensator(previous_features, reconstructed_offsets)

            reconstructed_residuals = self.residual_compressor.decompress(rp, rhp)
            reconstructed_features = reconstructed_residuals + aligned_features

            reconstructed_frame = self.reconstruction_head(reconstructed_features)
            decoded_frames.append(reconstructed_frame)
            previous_features = self.feature_extraction(reconstructed_frame)

        return torch.stack(decoded_frames).squeeze(1).unsqueeze(0)

    def forward(self, previous_lqf: torch.Tensor, current_lqf: torch.Tensor, current_hqf: torch.Tensor = None):
        # [SHARED] Get into feature space
        previous_features = self.feature_extraction(previous_lqf)
        current_features = self.feature_extraction(current_lqf)

        # [SHARED] Estimate motion, compress motion, decompress motion and align previous features
        offsets = self.motion_estimator(previous_features, current_features)
        decompressed_offsets, bits_offsets_prior, bits_offsets_hyperprior = (
            self.motion_compressor.train_compression_decompression(offsets)
        )
        aligned_features = self.motion_compensator(previous_features, decompressed_offsets)

        # [VC branch] Get residual, compress it and decompress
        residuals = current_features - aligned_features
        decompressed_residuals, bits_residuals_prior, bits_residuals_hyperprior = (
            self.residual_compressor.train_compression_decompression(residuals)
        )

        # [VC branch] Reconstruct features from residual and reconstruct frame
        reconstructed_features = decompressed_residuals + aligned_features
        reconstructed_frame = self.reconstruction_head(reconstructed_features)

        # [VC branch] Loss function
        aligned_frame = self.reconstruction_head(aligned_features)
        current_frame = self.reconstruction_head(current_features)
        frame_area = current_lqf.shape[0] * current_lqf.shape[2] * current_lqf.shape[3]  # B * H * W
        loss_vc = {
            "vc_recon_compression": self.rdr * self.reconstruction_loss(reconstructed_frame, current_lqf),
            "vc_recon_alignment": self.rdr * self.reconstruction_loss(aligned_frame, current_lqf),
            "vc_recon_features": self.rdr * self.reconstruction_loss(current_frame, current_lqf),
            "vc_bits_offsets_prior": bits_offsets_prior / frame_area,
            "vc_bits_offsets_hyperprior": bits_offsets_hyperprior / frame_area,
            "vc_bits_residuals_prior": bits_residuals_prior / frame_area,
            "vc_bits_residuals_hyperprior": bits_residuals_hyperprior / frame_area,
        }

        # [SR branch] Super-resolution upscale
        upscaled_frame = self.upscaler_head(aligned_features, current_lqf)

        # [SR branch] Loss function
        loss_vsr = {
            "vsr_recon": self.reconstruction_loss(upscaled_frame, current_hqf) if current_hqf is not None else None,
        }

        return VSRVCOutput(reconstructed_frame, upscaled_frame, loss_vc, loss_vsr)
