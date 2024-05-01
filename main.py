import importlib

import click
import torch.utils.data
from diffusers import UNet2DModel
import datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import RandomSampler
from torchvision.transforms import transforms

from diffustion import GaussianDiffusionDefault, make_beta_schedule


@click.command()
@click.option("--batch_size", default=1, type=int)
@click.option("--num_iters", default=5000, type=int)
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=int)
@click.option("--n_timestep", default=1000, type=int)
@click.option("--time_scale", default=1000, type=int)
@click.option("--lr", default=1.5e-5, type=float)
@click.option("--dataset", default="CelebaFolder", type=str)
@click.option("--dataset_path", default=r"D:\img_celeba", type=str)
def main(
    batch_size: int,
    num_iters: int,
    device: int,
    n_timestep: int,
    time_scale: int,
    lr: float,
    dataset: str,
    dataset_path: str,
):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
        ]
    )

    dataset_module = importlib.import_module(datasets.__name__)
    dataset_class = getattr(dataset_module, f"{dataset}Dataset")
    train_dataset = dataset_class(dataset_path, transform)
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=batch_size * num_iters)
    distill_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    total_steps = len(distill_train_loader)

    teacher_model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(
        device
    )
    student_model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(
        device
    )
    student_optimizer = AdamW(student_model.parameters(), lr=lr)
    scheduler = LinearLR(
        student_optimizer, start_factor=1, end_factor=0, total_iters=total_steps
    )
    teacher_betas = make_beta_schedule("cosine", n_timestep=n_timestep).to(device)
    teacher_ema_diffusion = GaussianDiffusionDefault(teacher_model, teacher_betas, time_scale)
    student_betas = make_beta_schedule(
        "cosine", n_timestep=teacher_ema_diffusion.num_timesteps // 2
    ).to(device)
    student_diffusion = GaussianDiffusionDefault(
        student_model,
        student_betas,
        teacher_ema_diffusion.time_scale * 2,
    )
    train_student(
        scheduler,
        student_optimizer,
        distill_train_loader,
        teacher_ema_diffusion,
        student_diffusion,
        student_model,
        device,
    )


if __name__ == "__main__":
    main()
    # main(
    #     batch_size=1,
    #     num_iters=5000,
    #     device=0 if torch.cuda.is_available() else "cpu",
    #     n_timestep=1000,
    #     time_scale=1000,
    #     lr=1.5e-5,
    #     celeba_path=r"D:\img_celeba",
    # )
