import random

from datasets import Vimeo90k
from metrics import psnr, ssim
from models.vsrvc import load_model
from utils import save_video, dict_to_string


def evaluate_model_e2e(chkpt_path: str, dataset_path: str, scale: int, examples: list = None, save_root: str = None):
    dataset = Vimeo90k(dataset_path, scale, test_mode=True)
    model = load_model(chkpt_path)
    if examples is None:
        examples = [random.randint(0, len(dataset))]

    for example in examples:
        lqs, hqs = dataset.__getitem__(example)
        lqs, hqs = lqs.to(model.device).unsqueeze(0), hqs.to(model.device).unsqueeze(0)
        data_root, bpp, upscaled = model.compress(lqs, keyframe_format="png")
        reconstructed = model.decompress(data_root)
        results = {"vc_psnr": psnr(reconstructed[0, 1:], lqs[0, 1:], aggregate="none"),
                   "vc_ssim": ssim(reconstructed[0, 1:], lqs[0, 1:], aggregate="none"),
                   "vsr_psnr": psnr(upscaled[0, 1:], hqs[0, 1:], aggregate="none"),
                   "vsr_ssim": ssim(upscaled[0, 1:], hqs[0, 1:], aggregate="none")}
        delimiter = "\n   "
        print(f"[{example}]:{delimiter}{dict_to_string(results, delimiter=delimiter)}")

        if save_root is not None:
            save_video(reconstructed, save_root, "evaluate_model_e2e_recon")
            save_video(upscaled, save_root, "evaluate_model_e2e_upscale")


if __name__ == "__main__":
    for chkpt in ["../outputs/1713122286.137639/model_10.pth", "../outputs/1713208917.474437/model_10.pth"]:
        evaluate_model_e2e(chkpt, "../../Datasets/VIMEO90k", 2, [791])
