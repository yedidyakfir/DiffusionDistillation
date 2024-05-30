import torch
import torchvision  # Import torchvision for saving images
from diffusers import DDPMPipeline, UNet2DModel
from diffusion2 import GaussianDiffusionDefault, make_beta_schedule
from tqdm import tqdm

def load_matching_keys(model, state_dict):
    """
    Loads matching keys from a given state_dict into the model.

    Args:
    - model (nn.Module): The model to load the state dictionary into.
    - state_dict (dict): The state dictionary containing model weights.

    Returns:
    None
    """
    model_state_dict = model.state_dict()
    matching_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
    model_state_dict.update(matching_state_dict)
    model.load_state_dict(model_state_dict)

def sample_images(model, diffusion_process, num_samples, device):
    """
    Generates samples from the given model using the diffusion process.

    Args:
    - model (nn.Module): The trained model to generate samples from.
    - diffusion_process (GaussianDiffusionDefault): The diffusion process to use for sampling.
    - num_samples (int): The number of samples to generate.
    - device (torch.device): The device to perform computations on (CPU or GPU).

    Returns:
    - samples (torch.Tensor): The generated samples.
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 3, 256, 256).to(device).float() # Generate random noise as starting point
        # print(f"x dtype: {x.dtype}") # debugging

        samples = diffusion_process.p_sample_loop(x, extra_args={}) # Perform sampling using the diffusion process
        # print(f"samples dtype: {samples.dtype}")  # debugging
    
    return samples

def main():
    """
    Main function to load models, perform sampling, and save the generated images.

    Args:
    None

    Returns:
    None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set device to GPU if available, otherwise CPU
    num_samples = 4 # Number of samples to generate
    checkpoint_path = 'checkpoint_829.pth' # Path to the checkpoint file

    # Load the pre-trained model and pipeline
    model_name = "google/ddpm-celebahq-256"
    pipeline = DDPMPipeline.from_pretrained(model_name).to(device)
    teacher_model = pipeline.unet # Get the UNet model from the pipeline

    # Initialize the student model with the same architecture as the teacher model
    student_model = UNet2DModel(**pipeline.unet.config).to(device)
    
    # Load the state dictionary of the student model from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student_model.load_state_dict(checkpoint['model_state_dict'])

    n_timestep = 512 # Number of timesteps for the diffusion process
    time_scale = 1 # Time scale for the diffusion process

    # Create beta schedule for the student model
    student_betas = make_beta_schedule("cosine", n_timestep=n_timestep).to(device).float()
    print(f"student_betas dtype: {student_betas.dtype}")

    # Initialize the Gaussian diffusion process for the student model
    student_diffusion = GaussianDiffusionDefault(student_model, student_betas, time_scale * 2)

    student_model = student_model.to(device).float()

    # Generate and save samples
    sample_images(student_model, student_diffusion, num_samples, device)

if __name__ == "__main__":
    main()
