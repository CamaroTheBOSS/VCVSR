import torch

from models.vsrvc import VSRVCModel
from models.vsrvc_no_alignment import VSRVCNoAlignment


def create_model_instance(model_type: str = "VSRVC", *args):
    if model_type == "VSRVC":
        return VSRVCModel(*args)
    elif model_type == "VSRVCNoAlignment":
        return VSRVCNoAlignment(*args)
    raise Exception(f"Unrecognized model type '{model_type}'")


def load_model(chkpt_path: str = None, name: str = "VSRVC", rdr: int = 128, vc: bool = True, vsr: bool = True,
               quant_type: str = "standard", model_type: str = "VSRVC"):
    if chkpt_path is None:
        return create_model_instance(model_type, name, rdr, vc, vsr, quant_type)
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
    model = create_model_instance(model_type, name, rdr, vc, vsr, quant_type)
    model.load_state_dict(saved_model['model_state_dict'])
    return model
