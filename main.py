import importlib
import click
import torch
import torch.utils.data
from diffusers import UNet2DModel, DDPMPipeline
import datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.utils.data import RandomSampler
from torchvision.transforms import transforms
from diffusion2 import GaussianDiffusionDefault, make_beta_schedule
from distillation import train_student

def load_matching_keys(model, state_dict):
    """
    Load state_dict into model, ignoring non-matching keys.
    
    Args:
        model (torch.nn.Module): The model into which to load the state dictionary.
        state_dict (dict): The state dictionary to load.
    """
    model_state_dict = model.state_dict()
    matching_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
    model_state_dict.update(matching_state_dict)
    model.load_state_dict(model_state_dict)

# CLI options using click for easy command-line interaction
@click.command()
@click.option("--batch_size", default=4, type=int, help="Batch size for training") # 4
@click.option("--num_iters", default=16, type=int, help="Number of progressive distillation iterations") # 64
@click.option("--device", default=0 if torch.cuda.is_available() else None, type=int, help="Device to use for training")
@click.option("--n_timestep", default=512, type=int, help="Number of timesteps for diffusion model")
@click.option("--time_scale", default=1, type=int, help="Time scale for the diffusion process")
@click.option("--lr", default=1.5e-5, type=float, help="Learning rate for the optimizer")
@click.option("--dataset", default="CelebaFolder", type=str, help="Dataset name to use for training")
@click.option("--dataset_path", default=r"img_celeba", type=str, help="Path to the dataset")

def main(batch_size: int, num_iters: int, device: int, n_timestep: int, time_scale: int, lr: float, dataset: str, dataset_path: str):
    """
    Main function to perform progressive distillation on the specified dataset.
    
    Args:
        batch_size (int): Size of each training batch.
        num_iters (int): Number of iterations for progressive distillation.
        device (int): Device index for training (0 for GPU, None for CPU).
        n_timestep (int): Number of timesteps in the diffusion model.
        time_scale (int): Scaling factor for the diffusion process.
        lr (float): Learning rate for the optimizer.
        dataset (str): Name of the dataset to use.
        dataset_path (str): Path to the dataset directory.
    """

    # Define the transformation pipeline for the dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Dynamically import the specified dataset module and class
    dataset_module = importlib.import_module(datasets.__name__)
    dataset_class = getattr(dataset_module, f"{dataset}Dataset")
    train_dataset = dataset_class(dataset_path, transform)

    # Use a subset of the dataset
    # subset_size = 100  # 100 images only for inference
    # indices = torch.randperm(len(train_dataset))[:subset_size]
    # subset = torch.utils.data.Subset(train_dataset, indices)


    # Create a data loader for the training dataset
    distill_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )

    # Initialize the teacher model and student model
    # teacher_model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)
    # student_model = UNet2DModel(in_channels=teacher_model.config.in_channels, out_channels=teacher_model.config.out_channels, layers_per_block=2, sample_size=256).to(device)
    # student_model = UNet2DModel.from_config(teacher_model.config).to(device)

    # Load pre-trained model
    model_name = "google/ddpm-celebahq-256"
    pipeline = DDPMPipeline.from_pretrained(model_name).to(device)
    teacher_model = pipeline.unet

    # Initialize student model with the same architecture
    student_model = UNet2DModel(**pipeline.unet.config).to(device)

    # Initialize optimizer and learning rate scheduler
    student_optimizer = AdamW(student_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(student_optimizer, T_max=num_iters)

    # Initialize diffusion process for the teacher model
    teacher_betas = make_beta_schedule("cosine", n_timestep=n_timestep).to(device)
    teacher_diffusion = GaussianDiffusionDefault(teacher_model, teacher_betas, time_scale)

    # Progressive Distillation Loop
    initial_sampling_steps = 100
    N = initial_sampling_steps

    # Calculate the maximum number of iterations
    max_possible_iterations = 0
    temp_N = N
    while temp_N >= 1:
        max_possible_iterations += 1
        temp_N //= 2

    # Adjust num_iters if it exceeds the max possible iterations
    num_iters = min(num_iters, max_possible_iterations)

    for iteration in range(num_iters):
        if N < 1:
            break  # Exit if N becomes less than 1

        student_betas = make_beta_schedule("cosine", n_timestep=N).to(device)
        student_diffusion = GaussianDiffusionDefault(student_model, student_betas, teacher_diffusion.time_scale * 2)

        train_student(
            scheduler,
            student_optimizer,
            distill_train_loader,
            teacher_diffusion,
            student_diffusion,
            student_model,
            device,
        )

        # Save student model checkpoint
        torch.save({
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': student_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': iteration,
            'sampling_steps': N,
        }, f'checkpoint_{iteration}.pth')

        # Update teacher model to the current student model's state
        # teacher_model.load_state_dict(student_model.state_dict())
        load_matching_keys(teacher_model, student_model.state_dict())


        # Halve the number of sampling steps
        N //= 2
        if N < 1:
            break  # Exit loop if N becomes less than 1

if __name__ == "__main__":
    main()
