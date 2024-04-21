import pathlib
import random

from datasets import Vimeo90k
from models.vsrvc import load_model
from utils import save_video


def evaluate_model_e2e(chkpt_path: str, dataset_path: str, scale: int, examples: list = None, save_root: str = None):
    dataset = Vimeo90k(dataset_path, scale, test_mode=True)
    model = load_model(chkpt_path)
    if examples is None:
        examples = [random.randint(0, len(dataset))]
    if save_root is None:
        save_root = "."

    for example in examples:
        lqs, hqs = dataset.__getitem__(example)
        data_root, bpp, upscaled = model.compress(lqs.unsqueeze(0), keyframe_format="png")
        reconstructed = model.decompress(data_root)
        save_video(reconstructed[0], save_root, name="evaluate_model_e2e_reconstruction")
        save_video(upscaled[0], save_root, name="evaluate_model_e2e_upscale")


if __name__ == "__main__":
    for chkpt in ["../outputs/1713122286.137639/model_10.pth", "../outputs/1713208917.474437/model_10.pth"]:
        parent = str(pathlib.Path(chkpt).parent)
        evaluate_model_e2e(chkpt, "../../Datasets/VIMEO90k", 2, [791], save_root=parent)
