import datetime
import glob
import math
import warnings
from collections import defaultdict, deque
from datetime import timedelta, datetime
from typing import Dict, Tuple

import logging
import shutil
from PIL import Image
import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler

import wandb
import time
import os


def save_video(video: torch.Tensor, root: str, name: str = "vid") -> None:
    path = os.path.join(root, name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    for i, frame in enumerate(video):
        numpy_frame = np.clip(frame.detach().permute(1, 2, 0).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        cv_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{path}/img{(i + 1)}.png", cv_frame)


def save_frame(filepath: str, frame: torch.Tensor):
    indexes = (0,) * (len(frame.shape) - 3) + (slice(None),)
    saved_frame = np.clip(frame[indexes].detach().cpu().permute(1, 2, 0).numpy() * 255., 0, 255).astype(np.uint8)
    saved_format = filepath[-3:]

    if saved_format == "png":
        cv2.imwrite(filepath, cv2.cvtColor(saved_frame, cv2.COLOR_RGB2BGR))
        return
    elif saved_format == "jpg":
        saved_frame = Image.fromarray(saved_frame)
        saved_frame.save(filepath, format="JPEG", quality=95)


def show_frame(window_name: str, frame: torch.Tensor):
    indexes = (0,) * (len(frame.shape) - 3) + (slice(None),)
    cv2_frame = np.clip(frame[indexes].detach().cpu().permute(1, 2, 0).numpy() * 255., 0, 255).astype(np.uint8)
    cv2.imshow(window_name, cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def add_dict(first: dict, second: Dict[str, torch.Tensor]):
    for key, value in second.items():
        if key in first.keys():
            first[key] += value.item()
        else:
            first[key] = value.item()
    return first


def dict_to_string(dictionary: dict, delimiter=", "):
    text = []
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            text.append(f"{key}: {value.tolist()}")
            continue
        text.append(f"{key}: {value}")
    return delimiter.join(text)


def init_run_dir(root: str, run_name: str = None) -> str:
    if run_name is None:
        run_name = str(datetime.now().timestamp())
    run_path = os.path.join(root, run_name)
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path)
    return run_path


def log_to_wandb(loss_dict: dict, metrics_dict: dict) -> None:
    wandb.log({"loss_" + key: value for key, value in loss_dict.items()}, commit=False)
    wandb.log({"metric_" + key: value for key, value in metrics_dict.items()})


def interpolate_video(video: torch.Tensor, size: Tuple[int, int], mode="bilinear"):
    if len(video.shape) == 4:
        video.unsqueeze(0)
    interpolated_video = torch.vmap(
        torch.nn.functional.interpolate, in_dims=(1,), out_dims=(1,)
    )(video, size=size, mode=mode, align_corners=False)
    return interpolated_video


def save_checkpoint(model: torch.nn.Module, root: str, epoch: int, only_last: bool = True):
    if only_last:
        pth_files = glob.glob(os.path.join(root, "model_*.pth"))
        for file in pth_files:
            os.remove(file)
    path = os.path.join(root, f"model_{epoch + 1}.pth")
    torch.save({'model_state_dict': model.state_dict(), 'model_name': "vcvsr"}, path)


class FileLogger:
    def __init__(self, log_directory: str, filename: str = "train.log") -> None:
        logging.basicConfig(format='%(message)s', filename=f"{log_directory}/train.log", filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.log_dir = f"{log_directory}/{filename}"
        self.init_log_file()

    def init_log_file(self) -> None:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            with open(self.log_dir, 'w') as f:
                self.logger.debug("")

    def log(self, lines) -> None:
        if isinstance(lines, str):
            lines = [lines]
        with open(self.log_dir, 'a+') as f:
            f.writelines('\n')
            for line in lines:
                self.logger.debug(line)
            self.logger.debug('\n' * 2)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.5f} ({global_avg:.5f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.force_save = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / (self.count + 0.00000001)

    @property
    def max(self):
        if len(self.deque):
            return max(self.deque)
        else:
            return 0

    @property
    def value(self):
        if len(self.deque):
            return self.deque[-1]
        else:
            return 0

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger:
    def __init__(self, delimiter="\t") -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.metrics = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.date_format = "%d-%m-%Y %H:%M:%S"

    def add_metric(self, name, metric) -> None:
        self.metrics[name] = metric

    def update(self, metrics: dict, prefix="") -> None:
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.metrics[prefix + key].update(value)

    def log(self, content, *args) -> None:
        for arg in args:
            content += str(arg)
        self.logger.info(content)

    def __getattr__(self, attr) -> SmoothedValue:
        if attr in self.metrics:
            return self.metrics[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.metrics.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def __call__(self, iterable, header="", print_every=1):
        idx = 0
        start_time = time.time()
        iteration_time = SmoothedValue(fmt="{avg:.4f}")
        current_index = ":" + str(len(str(len(iterable)))) + "d"

        log_data = ["[{date}]", header, "[{0" + current_index + "}/{1}]", "{eta}", "{iteration_time}/it", "{metrics}",
                    "GPU memory usage: {memory:.0f}MB" if torch.cuda.is_available() else "{memory}"]
        log_msg = self.delimiter.join(log_data)

        for obj in iterable:
            yield obj
            iteration_time.update(time.time() - start_time)
            eta_seconds = iteration_time.global_avg * (len(iterable) - 1)
            eta_string = str(timedelta(seconds=int(eta_seconds)))
            memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else "NAN"
            if idx % print_every == 0:
                self.log(log_msg.format(
                    idx,
                    len(iterable),
                    date=datetime.now().strftime(self.date_format),
                    eta=eta_string,
                    metrics=str(self),
                    iteration_time=str(iteration_time),
                    memory=memory, ))
            start_time = time.time()
            idx += 1


class MyCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose="deprecated", starting_point=0.):
        self.starting_step = int(T_max * starting_point)
        self.T_max = T_max - self.starting_step
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self._step_count  < self.starting_step:
            self.last_epoch = 0
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
