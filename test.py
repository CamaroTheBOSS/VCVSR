import torch


def my_quant(x: torch.Tensor):
    maxes = x.view(x.size(0), -1).max(dim=1).values
    mines = x.view(x.size(0), -1).min(dim=1).values
    zero_points = ((maxes + mines) / 2).view(-1, 1, 1)
    scales = ((maxes - mines) / 255).view(-1, 1, 1)
    ints_x = torch.round(((x - zero_points) / scales - 0.5)).to(torch.int8)
    float_x = ((ints_x.float() + 0.5) * scales) + zero_points
    noise = (torch.rand_like(x) - 0.5) * scales
    error_quant = ((float_x - x) ** 2).sum()
    error_noise = (noise ** 2).sum()
    print(error_quant, error_noise)


def uniform_quant(x: torch.Tensor):
    ints_x = torch.round(x).to(torch.int8)
    noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
    error_quant = ((ints_x - x) ** 2).sum()
    error_noise = (noise ** 2).sum()
    print(error_quant, error_noise)

def main():
    max_error = 0
    plus = (torch.rand(1) * 10 - 5).int().item()
    deviation = ((torch.rand(1)) * 25).item()
    x = ((torch.rand((16, 1024, 1024)) * 2 - 1) * deviation + plus)
    my_quant(x)
    uniform_quant(x)


if __name__ == "__main__":
    main()





