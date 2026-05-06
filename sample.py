import torch
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, UNet2DModel
import os
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import yaml
import datetime
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def range_constraint_loss(x):
    """
    Compute the range constraint loss: L_range = (1/N) * sum((x_i - clamp(x_i, -1, 1))^2)
    This is a soft constraint that encourages x to stay within [-1, 1].
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Range constraint loss
    """
    clamped_x = torch.clamp(x, -1, 1)
    loss = torch.mean((x - clamped_x) ** 2)
    return loss

def tv_regularization(x):
    """
    Compute the total variation regularization term.
    TV regularization encourages image smoothness and reduces noise.
    
    Args:
        x (torch.Tensor): Input tensor with shape (H, W) or (C, H, W)
        
    Returns:
        torch.Tensor: TV regularization loss
    """
    if x.dim() == 2:
        # 2D input: (H, W)
        tv_h = torch.mean(torch.abs(x[1:, :] - x[:-1, :]))
        tv_w = torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))
    elif x.dim() == 3:
        # 3D input: (C, H, W)
        tv_h = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")
    
    return tv_h + tv_w

def w2_distance_from_discretized_pdf(pdf_values1: torch.Tensor,
                                     pdf_values2: torch.Tensor,
                                     x_coords: torch.Tensor) -> torch.Tensor:
    """
    Compute the W2 distance between one or more pairs of discrete PDFs.
    This function is fully differentiable and supports batched multi-dimensional inputs.

    Args:
        pdf_values1 (torch.Tensor): PDF values for the first distribution.
                                    Supported shapes: (N,), (B, N), (B, R, N)
        pdf_values2 (torch.Tensor): PDF values for the second distribution. Must match pdf_values1.
        x_coords (torch.Tensor): x-coordinates corresponding to the PDF values. Shape: (N,)

    Returns:
        torch.Tensor: W2 distance.
                      If the input is (N,), returns a scalar.
                      If the input is (B, N), returns (B,).
                      If the input is (B, R, N), returns (B, R).
    """
    # Record the original shape so it can be restored later
    original_shape = pdf_values1.shape
    
    if original_shape != pdf_values2.shape:
        raise ValueError(f"Input tensor shapes must be identical. pdf1: {original_shape}, pdf2: {pdf_values2.shape}")
    
    if original_shape[-1] != x_coords.shape[0]:
        raise ValueError(f"The last dimension of input tensors ({original_shape[-1]}) must match the size of x_coords ({x_coords.shape[0]}).")

    # Reshape the inputs to (B_eff, N) for batching
    # B_eff is the product of all batch dimensions
    num_points = original_shape[-1]
    reshaped_pdf1 = pdf_values1.reshape(-1, num_points)
    reshaped_pdf2 = pdf_values2.reshape(-1, num_points)
    
    device = reshaped_pdf1.device
    effective_batch_size = reshaped_pdf1.shape[0]
    
    # --- 1. Make the PDF values non-negative and normalize them ---
    # Ensure non-negativity by subtracting the global minimum in each trace instead of taking abs
    min_pdf2 = reshaped_pdf2.min(dim=-1, keepdim=True)[0]
    
    non_negative_pdf1 = reshaped_pdf1 - min_pdf2*1.1
    non_negative_pdf2 = reshaped_pdf2 - min_pdf2*1.1

    # Helper function that computes the integral (total area) using the trapezoidal rule
    def _integrate(pdf, x):
        dx = torch.diff(x)
        # pdf shape: (B_eff, N), dx shape: (N-1)
        areas = (pdf[..., :-1] + pdf[..., 1:]) / 2.0 * dx
        return torch.sum(areas, dim=-1)

    total_area1 = _integrate(non_negative_pdf1, x_coords)
    total_area2 = _integrate(non_negative_pdf2, x_coords)

    # Normalize and add epsilon to avoid division by zero
    # total_area shapes are (B_eff,), need to be (B_eff, 1) for broadcasting
    reshaped_pdf1 = non_negative_pdf1 / (total_area1.unsqueeze(-1) + 1e-9)
    reshaped_pdf2 = non_negative_pdf2 / (total_area2.unsqueeze(-1) + 1e-9)
    
    # Helper function: convert a discrete PDF to a discrete CDF
    def _pdf_to_cdf(pdf_values, x):
        dx = torch.diff(x)
        areas = (pdf_values[..., :-1] + pdf_values[..., 1:]) / 2.0 * dx
        zeros = torch.zeros((effective_batch_size, 1), device=device, dtype=pdf_values.dtype)
        cdf_values = torch.cat([zeros, torch.cumsum(areas, dim=-1)], dim=-1)
        return cdf_values

    # Helper function: convert a discrete CDF to a discrete quantile function
    def _cdf_to_quantile(cdf_values, x, t):
        # cdf_values: (B_eff, N), t: (N,)
        # Expand t to (B_eff, N) so searchsorted handles batching correctly
        t_expanded = t.expand(cdf_values.shape[0], -1)
        right_indices = torch.searchsorted(cdf_values, t_expanded)
        
        right_indices = torch.clamp(right_indices, 1, num_points - 1)
        left_indices = right_indices - 1

        cdf_left = torch.gather(cdf_values, 1, left_indices)
        cdf_right = torch.gather(cdf_values, 1, right_indices)
        
        x_expanded = x.expand(effective_batch_size, -1)
        x_left = torch.gather(x_expanded, 1, left_indices)
        x_right = torch.gather(x_expanded, 1, right_indices)

        dcdf = cdf_right - cdf_left
        # Add epsilon to avoid division by zero
        dcdf = torch.where(dcdf < 1e-8, torch.tensor(1.0, device=device, dtype=dcdf.dtype), dcdf)
        slope = (x_right - x_left) / dcdf
        # Interpolate using the expanded t_expanded tensor
        quantile_values = x_left + (t_expanded - cdf_left) * slope
        
        return quantile_values

    # --- Main computation flow ---
    
    # 1. Compute the CDFs of both distributions
    cdf1 = _pdf_to_cdf(reshaped_pdf1, x_coords)
    cdf2 = _pdf_to_cdf(reshaped_pdf2, x_coords)

    # 2. Compute the quantile functions of both distributions
    t = torch.linspace(0, 1, num_points, device=device, dtype=reshaped_pdf1.dtype)
    
    quantile1 = _cdf_to_quantile(cdf1, x_coords, t)
    quantile2 = _cdf_to_quantile(cdf2, x_coords, t)

    # 3. Compute the W2 distance
    integrand = (quantile1 - quantile2) ** 2
    
    # Integrate along the last dimension (the N points)
    w2_squared = torch.trapezoid(integrand, t, dim=-1)
    
    result_flat = torch.sqrt(w2_squared) # Shape: (B_eff,)
    
    # Restore the original batch shape
    if len(original_shape) == 1: # Input was (N,)
        return result_flat.squeeze(0) # Return scalar
    else:
        output_shape = original_shape[:-1] # (B,) or (B, R)
        return result_flat.view(output_shape)

def agc_gain_control(wavelet, window_size=50, return_rms=False):
    """
    Automatic Gain Control (AGC) algorithm - vectorized implementation.
    Apply sliding-window gain control to each trace to balance energy.
    
    Args:
        wavelet (torch.Tensor): Input wavefield data with shape (n_shots, n_receivers, n_timesteps)
        window_size (int): Sliding window size
        return_rms (bool): Whether to return RMS values
        
    Returns:
        torch.Tensor: Wavefield data after AGC
        torch.Tensor (optional): RMS values with shape (n_shots, n_receivers, n_timesteps)
    """
    # Get the input shape
    n_shots, n_receivers, n_timesteps = wavelet.shape
    
    # Create the output tensor
    agc_wavelet = torch.zeros_like(wavelet)
    
    # Reshape the input to 2D for batching (n_shots*n_receivers, n_timesteps)
    wavelet_2d = wavelet.reshape(-1, n_timesteps)
    
    # Create a tensor to store RMS values
    rms_values = torch.zeros_like(wavelet_2d)
    # Compute the RMS value at each time step
    for t in range(n_timesteps):
        # Determine the window range
        start = max(0, t - window_size // 2)
        end = min(n_timesteps, t + window_size // 2)
        
        # Compute the RMS value within the window for all traces at the current time step
        window = wavelet_2d[:, start:end]
        rms = torch.sqrt(torch.mean(window ** 2, dim=1)) + 1e-9  # Add a small epsilon to avoid division by zero
        rms_values[:, t] = rms
    
    # Apply gain control
    agc_wavelet_2d = wavelet_2d / rms_values
    
    # Restore the original shape
    agc_wavelet = agc_wavelet_2d.reshape(n_shots, n_receivers, n_timesteps)
    rms_values = rms_values.reshape(n_shots, n_receivers, n_timesteps)
    
    if return_rms:
        return agc_wavelet, rms_values
    else:
        return agc_wavelet

def receiver(v, device, shot_num, source_locations='up', receiver_locations='up'):
    v = (torch.squeeze(v)+2)*1500

    dx = 10

    n_shots = shot_num

    n_sources_per_shot = 1
    d_source = int(70/n_shots) 
    first_source = 0  
    if source_locations == 'up' or source_locations == 'left':
        source_depth = 0
    elif source_locations == 'down'or source_locations == 'right':
        source_depth = 69
    else:
        raise ValueError("source_locations must be 'up', 'down', 'left', or 'right'")
    
    if receiver_locations == 'up' or receiver_locations == 'left':
        receiver_depth = 0
    elif receiver_locations == 'down' or receiver_locations == 'right':
        receiver_depth = 69
    else:
        raise ValueError("receiver_locations must be 'up', 'down', 'left', or 'right'")
    
    n_receivers_per_shot = 70
    d_receiver = 1 
    first_receiver = 0

    freq = 15
    nt = 1000
    dt = 0.001
    peak_time = 1.0933 / freq
    pml_width = [6,6,6,6]

    # source_locations
    if source_locations == 'up' or source_locations == 'down':
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                    dtype=torch.long, device=device)
        source_locations[..., 0] = source_depth
        source_locations[:, 0, 1] = (torch.arange(n_shots) * d_source +
                                    first_source)
    elif source_locations == 'left' or source_locations == 'right':
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                    dtype=torch.long, device=device)
        source_locations[..., 1] = source_depth
        source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                                    first_source)

    if receiver_locations == 'up' or receiver_locations == 'down':
        # receiver_locations
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                        dtype=torch.long, device=device)
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = (
            (torch.arange(n_receivers_per_shot) * d_receiver +
            first_receiver)
            .repeat(n_shots, 1)
        )
    elif receiver_locations == 'left' or receiver_locations == 'right':
        # receiver_locations
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                        dtype=torch.long, device=device)
        receiver_locations[..., 1] = receiver_depth
        receiver_locations[:, :, 0] = (
            (torch.arange(n_receivers_per_shot) * d_receiver +
            first_receiver)
            .repeat(n_shots, 1)
        )

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(n_shots, n_sources_per_shot, 1)
        .to(device)
    )

    out = scalar(v, dx, dt, source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=8,
                pml_width=pml_width,
                pml_freq=freq)

    receiver_amplitudes = out[-1]
    return receiver_amplitudes, source_locations, receiver_locations

def sample(scheduler, unet, npy_size, batch_size=1, sigma=0, rho=0, tau=1, gamma=1, save_x0_steps=False, x_true=None, loss_type='mse', shot_num=5, source_locations='up', receiver_locations='up', k=100, seed=1, normalize=True, adap_along=True):
    
    unet.eval()
    for param in unet.parameters():
        param.requires_grad = False
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(
        (batch_size, 1, npy_size, npy_size),
        generator=generator,
        device=device,
        dtype=unet.dtype
    )
    
    if save_x0_steps:
        x0_predictions = []
        
    # Initialize lists for storing intermediate results
    error_history = []
    error_0_history = []
    loss_history = []
    
    if x_true is not None:
        x_true = x_true.to(device)
        wave_true_init, source_locs_true, receiver_locs_true = receiver(x_true, device, shot_num=shot_num, source_locations=source_locations, receiver_locations=receiver_locations)
        # Fix the seed so the noise is reproducible
        wave_true_noisy = wave_true_init + sigma * torch.randn(
        wave_true_init.shape,
        generator=generator,
        device=device,
        dtype=unet.dtype
    )
        
        rms_values = wave_true_noisy
        rms_max = torch.max(abs(wave_true_noisy))

        weights = 1/(k*abs(rms_values)/rms_max+1)
        wave_true = wave_true_noisy * weights
        
    progress_bar = tqdm(scheduler.timesteps)
    
    for i, t_step in enumerate(progress_bar):
        latent_model_input = scheduler.scale_model_input(latents, t_step)
        if rho > 0:
            latent_model_input.requires_grad_(True)
        noise_pred = unet(latent_model_input, t_step).sample
            
        # Execute the scheduler step
        scheduler_output = scheduler.step(noise_pred, t_step, latents, generator=generator)
        latents = scheduler_output.prev_sample
        
        current_rho = rho
        if save_x0_steps:
            # Try to use pred_original_sample from the scheduler; fall back to manual computation if unsupported
            if hasattr(scheduler_output, 'pred_original_sample') and scheduler_output.pred_original_sample is not None:
                # Use the scheduler's pred_original_sample as the x_0 prediction
                x0_pred = scheduler_output.pred_original_sample
            else:
                # Manually compute the x_0 prediction
                # According to the DDPM formula: x_0 = (x_t - sqrt(1-alpha_t) * epsilon) / sqrt(alpha_t)
                # where epsilon is the noise prediction and alpha_t is the alpha value at the current timestep
                alpha_t = scheduler.alphas_cumprod[t_step]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

                x0_pred = (latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            if rho > 0:
                # Compute the loss according to the selected loss type
                wave_pred, source_locs_pred, receiver_locs_pred = receiver(x0_pred.squeeze()[1:-1, 1:-1], device, shot_num=shot_num, source_locations=source_locations, receiver_locations=receiver_locations)
                # Do not apply AGC to the synthetic wavefield; directly apply the observed wavefield weights to it
                wave_pred = wave_pred * weights

                # Compute the base loss (no longer using the muted wavefield)
                if loss_type == 'mse':
                    base_loss = torch.nn.functional.mse_loss(wave_pred, wave_true)
                elif loss_type == 'w2':
                    n_shots, n_receivers, n_timesteps = wave_pred.shape
                    wave_pred_reshaped = wave_pred.reshape(-1, n_timesteps)
                    wave_true_reshaped = wave_true.reshape(-1, n_timesteps)
                    x_coords = torch.linspace(0, 0.001 * (n_timesteps - 1), n_timesteps, device=device)
                    base_loss = torch.mean(w2_distance_from_discretized_pdf(wave_pred_reshaped, wave_true_reshaped, x_coords))
                    base_loss_normalizer = torch.mean(w2_distance_from_discretized_pdf(torch.zeros_like(wave_true_reshaped), wave_true_reshaped, x_coords))
                else:
                    raise ValueError("Unsupported loss type. Use 'mse', or 'w2'.")
                
                # Add the range constraint loss
                range_loss = range_constraint_loss(x0_pred.squeeze()[1:-1, 1:-1])
                
                # Add the TV regularization loss
                tv_loss = tv_regularization(x0_pred.squeeze()[1:-1, 1:-1])
                
                # Total loss = base loss + range constraint loss + TV regularization loss
                if normalize:
                    loss =  base_loss / base_loss_normalizer
                else:
                    loss =  base_loss

                weight = torch.autograd.grad(loss, x0_pred, retain_graph=True)[0]
                # print(weight.abs().mean(dim=3).squeeze())

                latents_grad_x0 = torch.autograd.grad(loss, x0_pred, retain_graph=True)[0]
                latents_grad = torch.autograd.grad(loss, latent_model_input)[0]
                
                c = 0.1
                depth = latents_grad.shape[-1]
                # Create a sequence from 1 to tau and repeat it depth times so each row is identical
                if gamma == 0:
                    change_rate = torch.ones_like(latents_grad_x0)
                else:
                    a = (tau-1)/(depth**gamma-1)
                    b = 1-a
                    base_rate = torch.linspace(1, tau, depth).unsqueeze(0).repeat(depth, 1).to(device)
                    # change_rate = a*base_rate**gamma+b
                    change_rate = ((torch.max(abs(latents_grad_x0))+tau)/(abs(latents_grad_x0)+tau))**gamma

                if adap_along:
                    latents = latents - current_rho * change_rate * latents_grad * torch.exp(-tv_loss/c)
                else:
                    latents = latents - current_rho * change_rate * latents_grad
                latents = latents.detach()
            
            x0_predictions.append(x0_pred.detach().cpu().squeeze().numpy())
        
        if x_true is not None:
            with torch.no_grad():
                error = torch.linalg.norm(latents.squeeze()[1:-1, 1:-1] - x_true)/torch.linalg.norm(x_true+2)
                error_0 = torch.linalg.norm(x0_pred.squeeze()[1:-1, 1:-1] - x_true)/torch.linalg.norm(x_true+2)
            if i == 0:
                print(f"Initial relative error: {error.item():.4f}, Initial x0 relative error: {error_0.item():.4f}")
            error_history.append(error.item())
            error_0_history.append(error_0.item())
            if rho > 0:
                loss_history.append(loss.item())
            else:
                loss_history.append(0.0)
                
            progress_bar.set_postfix(error=error.item(), error_0=error_0.item(), loss=loss.item() if rho > 0 else 0, current_rho=current_rho)
            
    if save_x0_steps:
        return latents, x0_predictions, error_history, error_0_history, loss_history, wave_true_noisy
    return latents, None, error_history, error_0_history, loss_history, wave_true_noisy

def plot_npy(latents, path, rows=4, cols=4, rho=0):
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        im = ax.imshow(latents[i, 0].detach().cpu().numpy())
        ax.axis("off")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"sample_rho_{rho}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)

def plot_real_and_pred(x_true, x_pred, path, rho=0):
    os.makedirs(path, exist_ok=True)
    # Get the data
    true_data = x_true.detach().cpu().numpy()
    pred_data = x_pred.detach().cpu().numpy()
    diff_data = true_data - pred_data

    # Compute the global minimum and maximum for the true and predicted data
    vmin = min(np.min(true_data), np.min(pred_data))
    vmax = max(np.max(true_data), np.max(pred_data))

    # Compute the color range for the difference plot
    diff_vmax = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
    diff_vmin = -diff_vmax

    # True plot
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(true_data, vmin=vmin, vmax=vmax)
    ax.set_title("True")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"real.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Predicted plot
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(pred_data, vmin=vmin, vmax=vmax)
    ax.set_title("Predicted")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"pred.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Difference plot
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(diff_data, vmin=diff_vmin, vmax=diff_vmax)
    ax.set_title("Difference")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"diff.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_wavefield(wavefield, path, filename_prefix, rho=0):
    os.makedirs(path, exist_ok=True)
    n_shots = wavefield.shape[0]
    plt.figure(figsize=(5, 5))

    i = 5  # Choose one trace for visualization
    ax = plt.gca()
    # Only plot the first 70 points using a grayscale imshow plot
    wavefield_np = wavefield[i].cpu().numpy().T  # Transpose and flip so the time axis is vertical and the receiver axis is horizontal
    data = wavefield_np[:, 0:70]
    # Invert the values so larger values become smaller and smaller values become larger
    data_inverted = data.max() + data.min() - data
    im = ax.imshow(data_inverted, cmap='gray', aspect='auto')
    ax.set_title(f"Shot {i}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{filename_prefix}_shot{i}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    
def plot_npy_steps(x0_predictions, path, rows=4, cols=4, rho=0):
    os.makedirs(path, exist_ok=True)
    num_steps = len(x0_predictions)
    if num_steps > rows * cols:
        jump = (num_steps // (rows * cols)) + 1
        x0_predictions = [x0_predictions[i] for i in range(0, num_steps, jump)]
    num_steps = len(x0_predictions)
    if len(x0_predictions[0].shape)==2:
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        for step in range(num_steps):
            ax = axes[step // cols, step % cols]
            im = ax.imshow(x0_predictions[step])
            ax.axis("off")
            fig.colorbar(im, ax=ax)
        for step in range(num_steps, rows * cols):
            ax = axes[step // cols, step % cols]
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"x0_step_{step}_rho_{rho}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        for batch_idx in range(len(x0_predictions[0])):
            fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
            for step in range(num_steps):
                ax = axes[step // cols, step % cols]
                im = ax.imshow(x0_predictions[step][batch_idx])
                ax.axis("off")
                fig.colorbar(im, ax=ax)
            for step in range(num_steps, rows * cols):
                ax = axes[step // cols, step % cols]
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"batch_{batch_idx}_x0_step_{step}_rho_{rho}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()

def plot_errors(error_history, error_0_history, path, rho=0):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    steps = range(len(error_history))
    plt.plot(steps, error_history, label='Error', color='blue')
    plt.plot(steps, error_0_history, label='Error_0', color='red')
    plt.xlabel('Step')
    plt.ylabel('Error Value')
    plt.title(f'Error and Error_0 over Steps (rho={rho})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"errors_rho_{rho}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_loss(loss_history, path, rho=0):
    """
    Plot the loss curve.
    """
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    steps = range(len(loss_history))
    plt.plot(steps, loss_history, label='Loss', color='green')
    plt.xlabel('Step')
    plt.ylabel('Loss Value')
    plt.title(f'Loss over Steps (rho={rho})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"loss_rho_{rho}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def save_results_to_file(error, error_0, loss, loss_type, rho, weight_power, filename="result.txt"):
    """
    Save the final values of error, error_0, and loss to a file in append mode.
    Include information about the optimization method, error metric, and relevant parameters.
    """
    with open(filename, "a") as f:
        f.write(f"Error metric: {loss_type}\n")
        f.write(f"Gradient descent step size (rho): {rho}\n")
        f.write(f"Weight power: {weight_power}\n")
        f.write(f"Final Error: {error}\n")
        f.write(f"Final Error_0: {error_0}\n")
        f.write(f"Final Loss: {loss}\n")
        f.write("-" * 50 + "\n")



def plot_wavefield_comparison(wave_true, wave_pred, path, filename_prefix, rho=0):
    os.makedirs(path, exist_ok=True)
    n_shots = wave_true.shape[0]
    
    fig, axes = plt.subplots(3, n_shots, figsize=(15, 9))
    if n_shots == 1:
        axes = axes.reshape(3, 1)

    # Use a shared colorbar range
    vmax = max(torch.max(torch.abs(wave_true)).item(), torch.max(torch.abs(wave_pred)).item())
    vmin = -vmax
    diff_vmax = torch.max(torch.abs(wave_true - wave_pred)).item()
    diff_vmin = -diff_vmax

    # Titles
    row_titles = ["Predicted", "True", "Difference"]

    for i in range(n_shots):
        # Predicted
        im1 = axes[0, i].imshow(wave_pred[i].cpu().detach().numpy().T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(im1, ax=axes[0, i])
        axes[0, i].set_title(f"Shot {i+1}")
        if i == 0:
            axes[0, i].set_ylabel(row_titles[0])

        # True
        im2 = axes[1, i].imshow(wave_true[i].cpu().detach().numpy().T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(im2, ax=axes[1, i])
        if i == 0:
            axes[1, i].set_ylabel(row_titles[1])

        # Difference
        im3 = axes[2, i].imshow((wave_true - wave_pred)[i].cpu().detach().numpy().T, cmap='seismic', aspect='auto', vmin=diff_vmin, vmax=diff_vmax)
        fig.colorbar(im3, ax=axes[2, i])
        if i == 0:
            axes[2, i].set_ylabel(row_titles[2])

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{filename_prefix}_rho_{rho}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_receiver_waveforms(wave_true, wave_pred, path, filename_prefix, receiver_idx=0, rho=0):
    """
    Plot 1D waveforms for all shots received by the specified receiver.
    Plot the true and predicted values on the same figure, without the difference.
    
    Args:
        wave_true: True wavefield data with shape (n_shots, n_receivers, n_timesteps)
        wave_pred: Predicted wavefield data with shape (n_shots, n_receivers, n_timesteps)
        path: Save path
        filename_prefix: File name prefix
        receiver_idx: Receiver index to plot, defaults to 0 (the first receiver)
        rho: Optimization parameter
    """
    os.makedirs(path, exist_ok=True)
    n_shots = wave_true.shape[0]
    n_timesteps = wave_true.shape[2]
    
    # Create subplots, one for each shot
    fig, axes = plt.subplots(n_shots, 1, figsize=(12, 3 * n_shots))
    if n_shots == 1:
        axes = [axes]  # If there is only one shot, ensure axes is list-like
    
    # Time axis
    time_axis = np.linspace(0, n_timesteps * 0.001, n_timesteps)  # Convert to seconds
    
    # Use a shared y-axis range
    y_max = max(torch.max(torch.abs(wave_true[:, receiver_idx, :])).item(),
                torch.max(torch.abs(wave_pred[:, receiver_idx, :])).item())
    
    for i in range(n_shots):
        ax = axes[i]
        
        # Extract data for the current shot and specified receiver
        true_data = wave_true[i, receiver_idx, :].cpu().detach().numpy()
        pred_data = wave_pred[i, receiver_idx, :].cpu().detach().numpy()
        
        # Plot the waveforms
        ax.plot(time_axis, true_data, 'b-', label='True', linewidth=1.5)
        ax.plot(time_axis, pred_data, 'r--', label='Predicted', linewidth=1.5)
        
        # Set the title and labels
        ax.set_title(f"Shot {i+1}, Receiver {receiver_idx+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set a shared y-axis range
        ax.set_ylim(-y_max * 1.1, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{filename_prefix}_receiver_{receiver_idx+1}_rho_{rho}.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_receiver_weights(weights, path, filename_prefix, receiver_idx=0, rho=0):
    """
    Plot the weights for all shots received by the specified receiver.
    Display the weights as a 1D line plot.
    
    Args:
        weights: Weight data with shape (n_shots, n_receivers, n_timesteps)
        path: Save path
        filename_prefix: File name prefix
        receiver_idx: Receiver index to plot, defaults to 0 (the first receiver)
        rho: Optimization parameter
    """
    os.makedirs(path, exist_ok=True)
    n_shots = weights.shape[0]
    n_timesteps = weights.shape[2]
    
    # Create subplots, one for each shot
    fig, axes = plt.subplots(n_shots, 1, figsize=(12, 3 * n_shots))
    if n_shots == 1:
        axes = [axes]  # If there is only one shot, ensure axes is list-like
    
    # Time axis
    time_axis = np.linspace(0, n_timesteps * 0.001, n_timesteps)  # Convert to seconds
    
    # Use a shared y-axis range
    y_min = torch.min(weights).item()
    y_max = torch.max(weights).item()
    
    for i in range(n_shots):
        ax = axes[i]
        
        # Extract weights data for the current shot and specified receiver
        weights_data = weights[i, receiver_idx, :].cpu().detach().numpy()
        
        # Plot weights as a 1D line chart
        ax.plot(time_axis, weights_data, 'g-', linewidth=1.5)
        
        # Set the title and labels
        ax.set_title(f"Shot {i+1}, Receiver {receiver_idx+1} Weights")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Weight Value")
        ax.grid(True, alpha=0.3)
        
        # Set a shared y-axis range
        ax.set_ylim(y_min * 0.9, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{filename_prefix}_receiver_{receiver_idx+1}_weights_rho_{rho}.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def save_latents_pt(x, path, rho):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(x, os.path.join(path, f"pred.pt"))

if __name__ == "__main__":
    image_size = 72
    batch_size = 1

    config_path = './configs/sample-config-FlatFault-B.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = config.get("model_path")

    unet = UNet2DModel.from_pretrained(model_path, subfolder="unet").to(device)
    
    receiver_locations = 'up'
    source_locations = 'up'

    ddpm_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    ddpm_scheduler.set_timesteps(1000)  # Set the number of sampling steps

    loss_type = config.get("loss_type", 'w2')
    testdata_path = config.get("testdata_path")
    x_true = np.load(testdata_path)
    x_true=torch.tensor(x_true, device=device, dtype=unet.dtype)/1500-2
    k = config.get("k", 100)
    seed = config.get("seed", 9)
    shot_num = config.get("shot_num", 10)
    rho = config.get("rho", 1.65)
    tau = config.get("tau", 1e-4)
    gamma = config.get("gamma", 0.55)
    normalize = config.get("normalize", True)
    adap_along = config.get("adap_along", True)
    sigma = config.get("sigma", 0)


    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = f"experiments/{now}"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    torch.manual_seed(42)
    np.random.seed(42)
        
    # Run sampling
    latents_ddpm, x0_predictions, error_history, error_0_history, loss_history, wave_true_noisy = sample(
        ddpm_scheduler, unet, image_size,
        batch_size=batch_size,
        sigma=sigma,
        rho=rho,
        save_x0_steps=True,
        x_true=x_true,
        loss_type=loss_type,
        shot_num=shot_num,
        tau=tau,
        gamma=gamma,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        k = k,  # Weight exponent p
        seed=seed,
        normalize=normalize,
        adap_along=adap_along
    )
    
    # plot_npy_steps(x0_predictions, path=os.path.join(target_path,f"./x0_steps_{loss_type}"), rho=rho, rows=8, cols=8)
    plot_real_and_pred(x_true, latents_ddpm.squeeze()[1:-1, 1:-1], path=os.path.join(target_path), rho=rho)
    


    # Plot the error and error_0 curves
    plot_errors(error_history, error_0_history, path=os.path.join(target_path), rho=rho)
    
    # Plot the loss curve
    plot_loss(loss_history, path=os.path.join(target_path), rho=rho)

    save_latents_pt(latents_ddpm.squeeze()[1:-1,1:-1], path=os.path.join(target_path, "data"), rho=rho)
    
    # Recompute the final wavefield for plotting
    x_pred_final = latents_ddpm.squeeze()[1:-1, 1:-1]
    wave_pred_final, source_locs_pred, receiver_locs_pred = receiver(x_pred_final, device, shot_num=shot_num, source_locations=source_locations, receiver_locations=receiver_locations)
    wave_true_final, source_locs_true, receiver_locs_true = receiver(x_true, device, shot_num=shot_num, source_locations=source_locations, receiver_locations=receiver_locations)
    
    # Plot 1D waveforms for the first receiver across five shots (before AGC)
    plot_receiver_waveforms(
        wave_true_final,
        wave_pred_final,
        path=os.path.join(target_path, ),
        filename_prefix="receiver_waveforms",
        receiver_idx=0,  # First receiver
        rho=rho
    )
    
    rms_values_final = wave_true_final
    rms_max_final = torch.max(abs(rms_values_final))
    weights_final = 1/(k*abs(rms_values_final)/rms_max_final+1)
    wave_true_final_agc = wave_true_final * weights_final
    wave_pred_final_agc = wave_pred_final * weights_final

    plot_wavefield(wave_true_noisy, path=os.path.join(target_path, ), filename_prefix="wavefield_noisy", rho=rho)
    
    # Plot 1D waveforms for the first receiver across five shots (after AGC)
    plot_receiver_waveforms(
        wave_true_final_agc,
        wave_pred_final_agc,
        path=os.path.join(target_path, ),
        filename_prefix="receiver_waveforms_agc",
        receiver_idx=0,  # First receiver
        rho=rho
    )
    
    # Plot weights for the first receiver across five shots
    plot_receiver_weights(
        weights_final,
        path=os.path.join(target_path, ),
        filename_prefix="receiver_weights",
        receiver_idx=0,  # First receiver
        rho=rho
    )

    shutil.copy(config_path, os.path.join(target_path, "sample_config.yaml"))
    
    # Save the final results to a file
    if error_history and error_0_history and loss_history:
        save_results_to_file(
            error_history[-1],
            error_0_history[-1],
            loss_history[-1],
            loss_type=loss_type,
            rho=rho,
            weight_power=k,
            filename=os.path.join(target_path,f"./result_{loss_type}.txt")
        )