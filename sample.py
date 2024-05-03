import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from diffustion import GaussianDiffusionDefault, make_beta_schedule
from diffusers import UNet2DModel

def load_model(model_path, device):
    net = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(
        device
    )  # Initialize your network class
    betas = make_beta_schedule("cosine", n_timestep=1000)  # Customize your beta schedule
    model = GaussianDiffusionDefault(net, betas.to(device), time_scale=1)#.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.net_.load_state_dict(checkpoint['model_state_dict'])  # Ensure net_ is an nn.Module and can handle .to(device)
    model.net_.to(device)  # Move only the neural network part to the device

    return model

def generate_sample(model, device):
    model.eval()
    with torch.no_grad():
        starting_noise = torch.randn([1, 3, 256, 256], device=device)  # Adjust dimensions 
        sample = model.p_sample_loop(starting_noise, extra_args={}, eta=0) 
        return sample

def save_and_show_sample(sample, filename):
    save_image(sample, filename)
    image = plt.imread(filename)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('checkpoint_9.pth', device)
    sample = generate_sample(model, device)
    save_and_show_sample(sample, 'generated_sample.png')
