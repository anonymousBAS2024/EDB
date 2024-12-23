import enum
import math
import random
import numpy as np
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

import torch.distributed as dist

from config.sisr.models.denoising_model import append_zero




class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

    
class EDB():
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, device=None, steps=10):
        super().__init__(device, steps)
        self._initialize(device, steps)

    def _initialize(self, device, rho=7.0, steps=10):
        self.noise_schedule = VENoiseSchedule(sigma_max = 80.0)
        self.ts = get_sigmas_karras(steps, 0.0001, 1 - 1e-4, rho, device=device)

        self.model = None


    #####################################

    # set lq for different cases
    def set_lq(self, lq):
        self.lq = lq

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    ####################################



    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t
    
    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t, b_t, c_t = (
            (alpha_bar_t * rho_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t * rho_t) / self.rho_T,
        )
        return a_t, b_t, c_t
    
    def get_f_g2(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2


    # diffusion process（forward process）
    def bridge_sample(self, x0, xT):
        noise = torch.randn_like(x0)
        x0 = x0.to(self.device)
        xT = xT.to(self.device)

        self.set_lq(xT)

        t = torch.randint(1, self.T)

        a_t, b_t, c_t = [self.append_dims(item, x0.ndim) for item in self.noise_schedule.get_abc(t)]
        samples = a_t * xT + b_t * x0 + c_t * noise
        return samples
    

    def append_dims(x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]


      
    @torch.no_grad()
    def sample_dbim(
        self,
        denoiser,
        diffusion,
        x,
        ts,
        eta=1.0,
        mask=None,
        **kwargs,
    ):
        x_T = x
        path = []
        pred_x0 = []

        ones = x.new_ones([x.shape[0]])
        indices = range(len(ts) - 1)
        indices = tqdm(indices, disable=(dist.get_rank() != 0))

        nfe = 0
        x0_hat = denoiser(x, diffusion.t_max * ones)
        torch.manual_seed(42)
        noise = torch.randn_like(x0_hat)
        first_noise = noise
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)
        x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

        for _, i in enumerate(indices):
            s = ts[i]
            t = ts[i + 1]

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)

            a_s, b_s, c_s = [self.append_dims(item, x0_hat.ndim) for item in self.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [self.append_dims(item, x0_hat.ndim) for item in self.noise_schedule.get_abc(t * ones)]

            _, _, rho_s, _ = [self.append_dims(item, x0_hat.ndim) for item in self.noise_schedule.get_alpha_rho(s * ones)]
            alpha_t, _, rho_t, _ = [
                self.append_dims(item, x0_hat.ndim) for item in self.noise_schedule.get_alpha_rho(t * ones)
            ]

            omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
            tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s

            noise = torch.randn_like(x0_hat)

            x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x + (1 if i != len(ts) - 2 else 0) * omega_st * noise

            path.append(x.detach().cpu())
            pred_x0.append(x0_hat.detach().cpu())
            nfe += 1

        return x, path, nfe, pred_x0, ts, first_noise


    def save_image(self, tensor, filename, cmap='gray'):
        import matplotlib.pyplot as plt
        import numpy as np

        tensor = tensor.cpu().squeeze().numpy() 

        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = np.transpose(tensor, (1, 2, 0))

        plt.imshow(tensor, cmap=cmap)
        plt.axis('off') 
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)



class NoiseSchedule:
    def __init__(self):
        raise NotImplementedError

    def get_f_g2(self, t):
        raise NotImplementedError

    def get_alpha_rho(self, t):
        raise NotImplementedError

    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t, b_t, c_t = (
            (alpha_bar_t * rho_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t * rho_t) / self.rho_T,
        )
        return a_t, b_t, c_t


class VPNoiseSchedule(NoiseSchedule):
    def __init__(self, beta_d=2, beta_min=0.1):
        self.beta_d, self.beta_min = beta_d, beta_min
        self.alpha_fn = lambda t: np.e ** (-0.5 * beta_min * t - 0.25 * beta_d * t**2)
        self.alpha_T = self.alpha_fn(1)
        self.rho_fn = lambda t: (np.e ** (beta_min * t + 0.5 * beta_d * t**2) - 1).sqrt()
        self.rho_T = self.rho_fn(torch.DoubleTensor([1])).item()

        self.f_fn = lambda t: (-0.5 * beta_min - 0.5 * beta_d * t)
        self.g2_fn = lambda t: (beta_min + beta_d * t)

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class VENoiseSchedule(NoiseSchedule):
    def __init__(self, sigma_max=80.0):
        self.sigma_max = sigma_max
        self.alpha_fn = lambda t: torch.ones_like(t)
        self.alpha_T = 1
        self.rho_fn = lambda t: t
        self.rho_T = sigma_max

        self.f_fn = lambda t: torch.zeros_like(t)
        self.g2_fn = lambda t: 2 * t

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


    

