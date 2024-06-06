import torch
from matplotlib import pyplot as plt


def norm_quant(x: torch.Tensor, mae=False):
    maxes = x.view(x.size(0), -1).max(dim=1).values
    mines = x.view(x.size(0), -1).min(dim=1).values
    zero_points = ((maxes + mines) / 2).view(-1, 1, 1)
    scales = ((maxes - mines) / 255).view(-1, 1, 1)
    ints_x = torch.round(((x - zero_points) / scales - 0.5)).to(torch.int8)
    float_x = ((ints_x.float() + 0.5) * scales) + zero_points
    noise = (torch.rand_like(x) - 0.5) * scales
    if mae:
        error_quant = torch.abs(float_x - x).sum()
        error_noise = torch.abs(noise).sum()
    else:
        error_quant = ((float_x - x) ** 2).sum()
        error_noise = (noise ** 2).sum()
    return error_quant, error_noise


def standard_quant(x: torch.Tensor, mae=False):
    ints_x = torch.round(x).to(torch.int8)
    noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
    if mae:
        error_quant = torch.abs(ints_x - x).sum()
        error_noise = torch.abs(noise).sum()
    else:
        error_quant = ((ints_x - x) ** 2).sum()
        error_noise = (noise ** 2).sum()
    return error_quant, error_noise


def main():
    stds = torch.linspace(0, 1, 1000)
    norm_quant_errors, std_quant_errors = [], []
    for std in stds:
        x = torch.normal(mean=0, std=std, size=(1, 128, 128))
        x = torch.clamp(x, -128, 127)
        norm_quant_error, norm_quant_noise_error = norm_quant(x, mae=True)
        std_quant_error, std_quant_noise_error = standard_quant(x, mae=True)
        norm_quant_errors.append(norm_quant_noise_error)
        std_quant_errors.append(std_quant_noise_error)
        # print((std, norm_quant_error, std_quant_error))
    plt.plot(stds, std_quant_errors)
    plt.plot(stds, norm_quant_errors)
    plt.xlabel("Odchylenie standardowe danych wejściowych")
    plt.ylabel("Błąd kwantyzacji MAE")
    plt.legend(["Standardowa kwantyzacja", "Znormalizowana kwantyzacja"])
    plt.show()


if __name__ == "__main__":
    main()





