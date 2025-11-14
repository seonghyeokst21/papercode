
import torch
import logging
import argparse
import numpy as np
import os
import shutil
import time
from torch.multiprocessing import Process
import torch.distributed as dist
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.nn import functional as F
import torch.nn as nn
from torch import optim
import torchvision
import pywt
import matplotlib.pyplot as plt

def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)

def adaptive_lambda(wavelet_loss_value, midpoint = 0.12, scale=0.05):
    return 1 / (1 + np.exp(-(wavelet_loss_value - midpoint) / scale))

def calculate_lambdas(wavelet_loss_syn_value, wavelet_loss_cyc_value, desired_sum=3):
    lambda_cGAN = adaptive_lambda(wavelet_loss_syn_value)
    lambda_cycleGAN = adaptive_lambda(wavelet_loss_cyc_value)

    lambda_cGAN_ratio = lambda_cGAN
    lambda_cycleGAN_ratio = 1 - lambda_cGAN_ratio

    min_val, max_val = 0.5, 2.5
    lambda_cGAN_normalized = lambda_cycleGAN_ratio * (max_val - min_val) + min_val
    lambda_cycleGAN_normalized = lambda_cGAN_ratio * (max_val - min_val) + min_val

    total = lambda_cGAN_normalized + lambda_cycleGAN_normalized
    scale_factor = desired_sum / total
    lambda_cGAN_normalized *= scale_factor
    lambda_cycleGAN_normalized *= scale_factor

    return lambda_cGAN_normalized, lambda_cycleGAN_normalized

def wavelet_loss(real_images, generated_images, wavelet='haar', level=1):
    """
    Calculate the wavelet loss between real and generated images.

    Args:
        real_images (list of torch.Tensor): List of real images tensors.
        generated_images (list of torch.Tensor): List of generated images tensors.
        wavelet (str): Type of wavelet to use.
        level (int): Level of decomposition.

    Returns:
        torch.Tensor: Wavelet loss value.
    """
    loss = 0.0
    for real, generated in zip(real_images, generated_images):
        # Detach tensors to avoid affecting gradient computation
        real_np = real.detach().cpu().numpy()
        generated_np = generated.detach().cpu().numpy()
        
        # Perform wavelet decomposition
        coeffs_real = pywt.wavedec2(real_np, wavelet=wavelet, level=level)
        coeffs_generated = pywt.wavedec2(generated_np, wavelet=wavelet, level=level)
        
        # Calculate high-frequency differences
        high_freq_real = coeffs_real[1:]  # Detail coefficients
        high_freq_generated = coeffs_generated[1:]
        
        # Sum of absolute differences in high-frequency components
        for hr, hg in zip(high_freq_real, high_freq_generated):
            for subband_r, subband_g in zip(hr, hg):
                # Convert numpy arrays back to tensors and compute loss
                subband_r_tensor = torch.from_numpy(subband_r).to(real.device)
                subband_g_tensor = torch.from_numpy(subband_g).to(real.device)
                loss += torch.mean(torch.abs(subband_r_tensor - subband_g_tensor))
    return loss

def save_high_freq_components(epoch, real_images, generated_images, wavelet='haar', level=1, save_path='./high_freq'):
    import os
    import numpy as np
    import pywt
    from matplotlib import pyplot as plt

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, (real_batch, generated_batch) in enumerate(zip(real_images, generated_images)):
        batch_size = real_batch.shape[0]
        for b in range(batch_size):
            real = real_batch[b]        # (channels, height, width)
            generated = generated_batch[b]  # (channels, height, width)

            # 채널 차원 처리 (그레이스케일 이미지를 가정)
            real_np = real.squeeze(0).detach().cpu().numpy()        # (height, width)
            generated_np = generated.squeeze(0).detach().cpu().numpy()  # (height, width)

            # 파이썬 고주파 성분 추출
            coeffs_real = pywt.wavedec2(real_np, wavelet=wavelet, level=level)
            coeffs_generated = pywt.wavedec2(generated_np, wavelet=wavelet, level=level)

            high_freq_real = coeffs_real[1:]      # Detail coefficients
            high_freq_generated = coeffs_generated[1:]  # Detail coefficients

            # 고주파 성분을 PNG 이미지로 저장
            for i, (hr, hg) in enumerate(zip(high_freq_real, high_freq_generated)):
                for j, (subband_r, subband_g) in enumerate(zip(hr, hg)):
                    subband_r_2d = subband_r
                    subband_g_2d = subband_g

                    # 실제 이미지 고주파 성분 저장
                    plt.imshow(subband_r_2d, cmap='gray')
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, f'epoch_{epoch}_real_{idx}_batch_{b}_level_{i}_subband_{j}.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # 생성된 이미지 고주파 성분 저장
                    plt.imshow(subband_g_2d, cmap='gray')
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, f'epoch_{epoch}_generated_{idx}_batch_{b}_level_{i}_subband_{j}.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()

def plot_and_save(epoch, wavelet_loss_syn_values, wavelet_loss_cyc_values, lambda_cGAN_values, lambda_cycleGAN_values, save_path):

    plt.figure(figsize=(12, 12))

    # Plotting Wavelet Loss Syn Histogram
    plt.subplot(2, 2, 1)
    plt.hist(wavelet_loss_syn_values, bins=30, color='blue', alpha=0.6)
    plt.title('Wavelet Loss Syn Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    mean_syn = np.mean(wavelet_loss_syn_values)
    std_syn = np.std(wavelet_loss_syn_values)
    plt.axvline(mean_syn, color='k', linestyle='dashed', linewidth=1)
    plt.text(mean_syn, plt.ylim()[1]*0.9, f'Mean: {mean_syn:.2f}\nStd: {std_syn:.2f}', ha='center')

    # Plotting Wavelet Loss Cyc Histogram
    plt.subplot(2, 2, 2)
    plt.hist(wavelet_loss_cyc_values, bins=30, color='orange', alpha=0.6)
    plt.title('Wavelet Loss Cyc Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    mean_cyc = np.mean(wavelet_loss_cyc_values)
    std_cyc = np.std(wavelet_loss_cyc_values)
    plt.axvline(mean_cyc, color='k', linestyle='dashed', linewidth=1)
    plt.text(mean_cyc, plt.ylim()[1]*0.9, f'Mean: {mean_cyc:.2f}\nStd: {std_cyc:.2f}', ha='center')

    # Plotting Lambda cGAN Histogram
    plt.subplot(2, 2, 3)
    plt.hist(lambda_cGAN_values, bins=30, color='green', alpha=0.6)
    plt.title('Lambda cGAN Distribution')
    plt.xlabel('Lambda Value')
    plt.ylabel('Frequency')
    mean_cgan = np.mean(lambda_cGAN_values)
    std_cgan = np.std(lambda_cGAN_values)
    plt.axvline(mean_cgan, color='k', linestyle='dashed', linewidth=1)
    plt.text(mean_cgan, plt.ylim()[1]*0.9, f'Mean: {mean_cgan:.2f}\nStd: {std_cgan:.2f}', ha='center')

    # Plotting Lambda cycleGAN Histogram
    plt.subplot(2, 2, 4)
    plt.hist(lambda_cycleGAN_values, bins=30, color='red', alpha=0.6)
    plt.title('Lambda cycleGAN Distribution')
    plt.xlabel('Lambda Value')
    plt.ylabel('Frequency')
    mean_cycgan = np.mean(lambda_cycleGAN_values)
    std_cycgan = np.std(lambda_cycleGAN_values)
    plt.axvline(mean_cycgan, color='k', linestyle='dashed', linewidth=1)
    plt.text(mean_cycgan, plt.ylim()[1]*0.9, f'Mean: {mean_cycgan:.2f}\nStd: {std_cycgan:.2f}', ha='center')

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'epoch_{epoch}_loss_lambda_histogram.png'))
    plt.close()

    # Print mean and standard deviation for each distribution
    print(f"Wavelet Loss Syn: Mean = {mean_syn:.2f}, Std = {std_syn:.2f}")
    print(f"Wavelet Loss Cyc: Mean = {mean_cyc:.2f}, Std = {std_cyc:.2f}")
    print(f"Lambda cGAN: Mean = {mean_cgan:.2f}, Std = {std_cgan:.2f}")
    print(f"Lambda cycleGAN: Mean = {mean_cycgan:.2f}, Std = {std_cycgan:.2f}")



### ### ### ### ### files & params ### ### ### ### ###
def copy_source(file, output_dir):
    """
    Copies a specified file to a given output directory.

    Args:
        file (str): Path of the file to be copied.
        output_dir (str): Path of the destination directory where the file will be copied.

    Returns:
        None
    """
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
    # File copy and 
            
def broadcast_params(params):
    """
    Broadcasts parameters to all nodes in a distributed computing setup.

    Args:
        params (list): A list of parameters (usually tensors) to be broadcasted.

    Returns:
        None
    """
    for param in params:
        dist.broadcast(param.data, src=0)

### ### ### ### ### Diffusion coefficients ### ### ### ### ###
def var_func_vp(t, beta_min, beta_max):
    """
    Calculates the variance using a specific variance function based on time steps and beta parameters.

    Args:
        t (Tensor): Time steps tensor.
        beta_min (float): Minimum beta value.
        beta_max (float): Maximum beta value.

    Returns:
        Tensor: Calculated variance.
    """
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    """
    Calculates the geometric variance based on time steps and beta parameters.

    Args:
        t (Tensor): Time steps tensor.
        beta_min (float): Minimum beta value.
        beta_max (float): Maximum beta value.

    Returns:
        Tensor: Geometric variance.
    """
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    """
    Extracts and reshapes a tensor based on provided time steps and shape.

    Args:
        input (Tensor): Input tensor.
        t (Tensor): Time steps tensor.
        shape (list): Target shape for reshaping.

    Returns:
        Tensor: Reshaped tensor.
    """
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    """
    Generates a time schedule tensor based on input arguments and specified device.

    Args:
        args: Argument object containing num_timesteps.
        device: Computational device (CPU/GPU). => Generally, "GPU"

    Returns:
        Tensor: Time schedule tensor.
    """
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    """
    Generates a sigma schedule based on input arguments and specified device.

    Args:
        args: Argument object containing num_timesteps, beta_min, beta_max, and use_geometric.
        device: Computational device (CPU/GPU).

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Tuple containing sigmas, a_s, and betas tensors.
    """
    n_timestep = args.num_timesteps
    beta_min = args.beta_min 
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

########
class Diffusion_Coefficients():
    """
    A class to hold and compute diffusion coefficients for the diffusion model.

    Attributes:
        sigmas (Tensor): A tensor of sigma values for each time step.
        a_s (Tensor): A tensor of 'a' values for each time step.
        a_s_cum (Tensor): Cumulative product of 'a' values.
        sigmas_cum (Tensor): Square root of (1 - cumulative square of 'a' values).
        a_s_prev (Tensor): A clone of 'a_s' with the last element set to 1.
    """

    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Applies diffusion process to the input data.

    Args:
        coeff (Diffusion_Coefficients): The diffusion coefficients.
        x_start (Tensor): The original data (at time t=0).
        t (Tensor): The current time step.
        noise (Tensor, optional): External noise. If None, random noise is generated.

    Returns:
        Tensor: The diffused data at time step t.
    """

    if noise is None:
      noise = torch.randn_like(x_start)
    #print('x_t-','a_s_cum',coeff.a_s_cum,'sigmas_cum', coeff.sigmas_cum, flush=True)
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generates a pair of disturbed images at time steps t and t+1 for training.

    Args:
        coeff (Diffusion_Coefficients): The diffusion coefficients.
        x_start (Tensor): The original data (at time t=0).
        t (Tensor): The current time step.

    Returns:
        tuple: A tuple containing disturbed images at time steps t and t+1.
    """

    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    
    #print('x_t_plus_one-','a_s',coeff.a_s,'sigmas', coeff.sigmas, flush=True)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one # it means the output is x_t, x_{t+1}

#%% posterior sampling
class Posterior_Coefficients():
    """
    The Posterior_Coefficients class calculates coefficients necessary for posterior sampling.

    Args:
        args: Object containing arguments, mainly settings for the diffusion schedule.
        device: The device (CPU/GPU) on which computations will be performed.

    Attributes:
        betas: Beta values according to the diffusion schedule.
        alphas: Array of values computed as 1 minus betas.
        alphas_cumprod: Cumulative product of alpha values.
        alphas_cumprod_prev: Previous values of the cumulative product of alphas.
        posterior_variance: Posterior variance values.
        sqrt_alphas_cumprod: Square root of the cumulative product of alphas.
        sqrt_recip_alphas_cumprod: Square root of the reciprocal of the cumulative product of alphas.
        sqrt_recipm1_alphas_cumprod: Square root of (reciprocal of the cumulative product of alphas minus 1).
        posterior_mean_coef1, posterior_mean_coef2: Coefficients used for calculating the posterior mean.
        posterior_log_variance_clipped: Logarithm of the clipped posterior variance.
    """

    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0, x_t, t):
    """
    Performs posterior sampling for given x_0, x_t, and t values.

    Args:
        coefficients: An instance of the Posterior_Coefficients class.
        x_0: Original data.
        x_t: Data after undergoing the diffusion process.
        t: Current time step.

    Returns:
        Tensor: Data generated via posterior sampling.
    """

    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    """
    Samples for n_time iterations using the provided generator model.

    Args:
        coefficients: An instance of the Posterior_Coefficients class.
        generator: Model for generating data.
        n_time: Number of time steps to iterate.
        x_init: Initial data.
        T: Time schedule.
        opt: Options object.

    Returns:
        Tensor: Generated data.
    """
    x = x_init[:,[0],:]
    source = x_init[:,[1],:]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,[0],:], x, t)
            x = x_new.detach()
        
    return x
