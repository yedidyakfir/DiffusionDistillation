import abc
import math
import torch
import torch.nn.functional as F
from torch.nn import Module
from tqdm import tqdm
import torchvision

def make_beta_schedule(schedule, n_timestep: int, cosine_s: float = 8e-3):
    """
    Creates a schedule for beta values used in the diffusion process.

    Args:
        schedule (str): Type of schedule to create ('cosine' is currently supported).
        n_timestep (int): Number of timesteps.
        cosine_s (float): Scaling factor for the cosine schedule.

    Returns:
        torch.Tensor: Tensor containing beta values for each timestep.
    """
    if schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise Exception("Unsupported schedule type")
    return betas

def E_(input, t, shape):
    """
    Gathers elements from input tensor based on the given timestep t.

    Args:
        input (torch.Tensor): Input tensor.
        t (torch.Tensor): Timestep tensor.
        shape (tuple): Shape of the output tensor.

    Returns:
        torch.Tensor: Gathered tensor reshaped to the specified shape.
    """
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

class GaussianDiffusion(Module):
    def __init__(self, net: Module, betas, time_scale: int = 1, sampler="ddpm"):
        """
        Initializes the Gaussian Diffusion model.

        Args:
            net (Module): Neural network model.
            betas (torch.Tensor): Beta schedule.
            time_scale (int): Scaling factor for time.
            sampler (str): Sampling method.
        """
        super().__init__()
        self.net_ = net
        self.time_scale = time_scale
        betas = betas.type(torch.float64)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64, device=betas.device), alphas_cumprod[:-1]),
            0,
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.posterior_variance = posterior_variance
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the diffusion process at a given timestep t.

        Args:
            x_start (torch.Tensor): Starting tensor.
            t (torch.Tensor): Timestep tensor.
            noise (torch.Tensor): Optional noise tensor.

        Returns:
            torch.Tensor: Sampled tensor.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            E_(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            E_(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        """
        Computes the loss for the diffusion model.

        Args:
            x_start (torch.Tensor): Starting tensor.
            t (torch.Tensor): Timestep tensor.
            noise (torch.Tensor): Optional noise tensor.

        Returns:
            torch.Tensor: Computed loss.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.net_(x_noisy, t / self.num_timesteps)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, extra_args, eta=0):
        """
        Sample a tensor from the posterior distribution.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Timestep tensor.
            extra_args (dict): Extra arguments for the model.
            eta (float): Optional parameter for controlling the amount of noise.

        Returns:
            torch.Tensor: Sampled tensor.
        """
        # print(f"p_sample: x dtype: {x.dtype}, t dtype: {t.dtype}")  # Debugging statement
        noise = torch.randn_like(x)
        t_scaled = t.float() / float(self.num_timesteps)
        x_recon = self.net_(x, t_scaled, **extra_args).sample
        # print(f"p_sample: x_recon dtype: {x_recon.dtype}")  # Debugging statement
        alpha = E_(self.sqrt_alphas_cumprod, t.long(), x.shape).float()
        sigma = E_(self.sqrt_one_minus_alphas_cumprod, t.long(), x.shape).float()
        return alpha * x_recon + sigma * noise

    @torch.no_grad()
    def p_sample_loop(self, x, extra_args, eta=0):
        """
        Loop over timesteps to sample from the posterior distribution.

        Args:
            x (torch.Tensor): Input tensor.
            extra_args (dict): Extra arguments for the model.
            eta (float): Optional parameter for controlling the amount of noise.

        Returns:
            torch.Tensor: Sampled tensor.
        """
        mode = self.net_.training
        self.net_.eval()
        for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
            t = torch.full((x.shape[0],), i, dtype=torch.int64).to(x.device)  # Ensure t is int64 for torch.gather
            # print(f"p_sample_loop: t dtype: {t.dtype}")  # Debugging statement
            x = self.p_sample(x, t, extra_args, eta=eta)
            # print(f"p_sample_loop: x dtype after p_sample: {x.dtype}")  # Debugging statement
            if i % 50 == 0:  # Save image every 50 steps
                self.save_image(x, i)
        self.net_.train(mode)
        return x

    def save_image(self, images, step):
        images = images.detach().cpu()
        grid = torchvision.utils.make_grid(images, nrow=4)
        torchvision.utils.save_image(grid, f"sample_step_{step}.png")

    def get_alpha_sigma(self, x, t):
        """
        Get alpha and sigma values for a given timestep t.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Timestep tensor.

        Returns:
            tuple: Alpha and sigma tensors.
        """
        alpha = E_(self.sqrt_alphas_cumprod, t, x.shape)
        sigma = E_(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha, sigma

    def inference(self, x, t, extra_args):
        """
        Perform inference using the model.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Timestep tensor.
            extra_args (dict): Extra arguments for the model.

        Returns:
            torch.Tensor: Output of the model.
        """
        return self.net_(x, t * self.time_scale, **extra_args).sample

class GaussianDiffusionDefault(GaussianDiffusion):
    def __init__(self, net: Module, betas: int, time_scale: int = 1, gamma: float = 0.3):
        super().__init__(net, betas, time_scale)
        self.gamma = gamma

    def distill_loss(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
        """
        Computes the distillation loss for training the student model.

        Args:
            student_diffusion (GaussianDiffusion): Student diffusion model.
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Timestep tensor.
            extra_args (dict): Extra arguments for the model.
            eps (torch.Tensor): Optional noise tensor.
            student_device (torch.device): Device for the student model.

        Returns:
            torch.Tensor: Computed distillation loss.
        """
        if eps is None:
            eps = torch.randn_like(x)
        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1)
            z = alpha * x + sigma * eps
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            v = self.inference(z.float(), t.float() + 1, extra_args).double()
            rec = (alpha * z - sigma * v).clip(-1, 1)
            z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
            v_1 = self.inference(z_1.float(), t.float(), extra_args).double()
            x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1)
            eps_2 = (z - alpha_s * x_2) / sigma_s
            v_2 = alpha_s * eps_2 - sigma_s * x_2
            if self.gamma == 0:
                w = 1
            else:
                w = torch.pow(1 + alpha_s / sigma_s, self.gamma)
        v = student_diffusion.net_(z.float(), t.float() * self.time_scale, **extra_args).sample
        return F.mse_loss(w * v.float(), w * v_2.float())
