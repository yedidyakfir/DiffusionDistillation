import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion import GaussianDiffusion

def moving_average(model, model_ema, beta=0.999, device=None):
    """
    Updates the parameters of model_ema using the moving average of the parameters of model.

    Args:
        model (Module): The model whose parameters will be averaged.
        model_ema (Module): The model to update with the averaged parameters.
        beta (float): The decay rate for the moving average.
        device (torch.device): The device to use for computation (optional).
    """
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
    """
    Trains the student model using progressive distillation.

    Args:
        scheduler (LRScheduler): Learning rate scheduler.
        optimizer (Optimizer): Optimizer for training.
        distill_train_loader (DataLoader): DataLoader for the training dataset.
        teacher_diffusion (GaussianDiffusion): Teacher diffusion model.
        student_diffusion (GaussianDiffusion): Student diffusion model.
        student_ema (Module): Exponential moving average of the student model.
        device (int): Device to use for training.
    """
    total_steps = len(distill_train_loader)
    teacher_diffusion.net_.eval()  # Set teacher model to evaluation mode
    student_diffusion.net_.train()  # Set student model to training mode
    pbar = tqdm(distill_train_loader)
    n = 0
    loss_total = 0
    for img, label in pbar:
        optimizer.zero_grad()
        img = img.to(device)
        num_timesteps = student_diffusion.num_timesteps
        if num_timesteps > 0:
            time = 2 * torch.randint(0, num_timesteps, (img.shape[0],), device=device)
        else:
            print(f"Skipping iteration: num_timesteps={num_timesteps} is not valid.")
            continue
        loss = teacher_diffusion.distill_loss(student_diffusion, img, time, {})
        loss_total += loss.item()
        n += 1
        pbar.set_description(f"Loss: {loss_total / n}")
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(student_diffusion.net_.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        moving_average(student_diffusion.net_, student_ema)
        
        # Save model checkpoint periodically
        if n % 100 == 0 or n == total_steps - 1:
            torch.save({
                'model_state_dict': student_diffusion.net_.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': n
            }, f'checkpoint_{n}.pth')

        if n > total_steps:
            break
