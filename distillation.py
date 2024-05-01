import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffustion import GaussianDiffusion


def moving_average(model, model_ema, beta=0.999, device=None):
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            data = param.data
            if device is not None:
                data = data.to(device)
            param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def train_student(
    scheduler: LRScheduler,
    optimizer: Optimizer,
    distill_train_loader: DataLoader,
    teacher_diffusion: GaussianDiffusion,
    student_diffusion: GaussianDiffusion,
    student_ema: Module,
    device: int,
):
    total_steps = len(distill_train_loader)
    teacher_diffusion.net_.eval()
    student_diffusion.net_.train()
    pbar = tqdm(distill_train_loader)
    n = 0
    loss_total = 0
    for img, label in pbar:
        optimizer.zero_grad()
        img = img.to(device)
        time = 2 * torch.randint(
            0, student_diffusion.num_timesteps, (img.shape[0],), device=device
        )
        loss = teacher_diffusion.distill_loss(student_diffusion, img, time, {})
        loss_total += loss.item()
        n += 1
        pbar.set_description(f"Loss: {loss_total / n}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        moving_average(student_diffusion.net_, student_ema)
        if n > total_steps:
            break
    return student_diffusion
