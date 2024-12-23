import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel
import matplotlib.pyplot as plt

from torchvision.transforms import transforms


logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)

        self.load()
        self.t_max = 1

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()


    
    def save_images(self,images, title, filename):
        """
        Save a grid of images to a file.
        
        Args:
            images (torch.Tensor): A tensor of shape [N, C, H, W] where N is the number of images.
            title (str): Title for the grid of images.
            filename (str): Path to save the output image.
        """
        # Convert tensor to NumPy array
        images = images.cpu().detach().numpy()
        
        # Create a grid of images
        N = images.shape[0]  # Number of images
        C, H, W = images.shape[1], images.shape[2], images.shape[3]
        
        fig, axes = plt.subplots(1, N, figsize=(N*4, 4))
        for i in range(N):
            ax = axes[i]
            img = images[i].transpose(1, 2, 0)  # Change to HWC format
            if C == 1:
                img = img.squeeze(-1)  # Remove the channel dimension if grayscale
            ax.imshow(img)
            ax.axis('off')
        plt.suptitle(title)
        plt.savefig(filename)
        plt.close()

    def sample(self, batch_size, device, t_max=None, t_min=None):
        if t_max==None and t_min==None:
            t_max = self.t_max 
            t_min = self.t_min
        ts = torch.rand(batch_size).to(device) * (t_max - t_min) + t_min
        return ts, torch.ones_like(ts)


    def feed_data(self, LQ, GT=None):
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
    
    
    def training_bridge_losses(self, current_step, model, x_start, model_kwargs=None, noise=None):
        assert model_kwargs is not None
        xT = model_kwargs["xT"]
        mask = model_kwargs.pop("mask", None)
        if noise is None:
            noise = torch.randn_like(x_start)

        t,_ = self.sample(xT.shape[0],self.device)
        t = torch.minimum(t, torch.ones_like(t) * self.t_max)
        terms = {}
        if current_step == 0:
            hat_x0 = x_start
        else:
            x_t = model.bridge_sample(x_start, xT, t, noise)

            _, hat_x0, weights = self.denoise(model, x_t, t, **model_kwargs)

        # adopt the estimated x_0 based on the learned parameters of last iteration

        t_2,_ = self.sample(xT.shape[0],self.device,t)
        t_2 = torch.minimum(t, torch.ones_like(t) * self.t_max)

        x_t_2 = model.bridge_sample(hat_x0, xT, t_2, noise)

        _, denoised, weights = self.denoise(model, x_t_2, t_2, **model_kwargs)

        if mask is not None:
            terms["xs_mse"] = mean_flat(mask * (denoised - x_start) ** 2)
            terms["mse"] = mean_flat(weights * mask * (denoised - x_start) ** 2)
        else:
            terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
            terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        terms["loss"] = terms["mse"]

        return terms
    


    def optimize_parameters_x0_ours(self, step, timesteps, sde=None):

        sde.set_lq(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)
        t_1 = timesteps.squeeze()
        batch = self.state.shape[0]

        t_2 = torch.zeros_like(t_1)
        for i in range(batch):
            upper_limit = t_1[i].item() + 1
            if upper_limit > 1:  
                t_2[i] = torch.randint(1, upper_limit, (1,)).long()
            else:
                t_2[i] = 1  

        timesteps_2 = t_2.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(self.device)

        x0_bar = sde.noise_fn(self.state, timesteps.squeeze())
        state_2, _ = sde.q_sample(x0_bar, timesteps_2)

        x0_bar_bar = sde.noise_fn(state_2, timesteps_2.squeeze())


        loss = self.loss_fn(x0_bar, self.state_0) +  self.loss_fn(x0_bar_bar, self.state_0)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        self.log_dict["loss"] = loss.item()


    # def test_visual_x0(self, current_step=-1, sde=None, save_states=False):
    #     sde.set_lq(self.condition)

    #     self.model.eval()
    #     with torch.no_grad():
    #         self.output = sde.reverse_sde_visual_x0(self.state, current_step=current_step, save_states=save_states)
    #     self.model.train()


    def test_visual_x0(self, current_step=-1, sde=None, save_states=False, name=None,t=None,skip=True):
        sde.set_lq(self.condition)

        self.model.eval()
        with torch.no_grad():
            if skip == False:
                self.output = sde.reverse_sde_visual_x0(self.state, current_step=current_step, save_states=save_states)
            # 指定输出某一步预测的x0
            else:
                if t==None:
                    t_expectation = 100
                else:
                    t_expectation = t
                self.output = sde.reverse_single_x0(self.state,t_expectation=t_expectation)

        self.model.train()
    
    def denoise(self, model, x_t, t, **model_kwargs):
        c_skip, c_in, c_out, c_noise, weightings = self.precond.get_scalings_and_weightings(t, x_t.ndim)
        model_output = model(c_in * x_t, c_noise, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised, weightings
    
    def denoiser(self, sde, x_t, sigma, **model_kwargs):
        _, denoised, _ = self.denoise(sde, x_t, sigma, **model_kwargs)
        denoised = denoised.clamp(-1, 1)
        return denoised
    
    # EDB test
    def test(self, sde=None):
        sde.set_lq(self.condition)

        self.model.eval()
        with torch.no_grad():
            self.output, _, _, _, _, _ = sde.sample_dbim(
                self.denoiser,
                sde,
                self.condition, #x_T
                sde.ts,
                eta=1.0,
                mask=None,
            )

        self.model.train()



    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')



def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class PreCond:
    def __init__(self, ns):
        raise NotImplementedError

    def _get_scalings_and_weightings(self, t):
        raise NotImplementedError

    def get_scalings_and_weightings(self, t, ndim):
        c_skip, c_in, c_out, c_noise, weightings = self._get_scalings_and_weightings(t)
        c_skip, c_in, c_out, weightings = [append_dims(item, ndim) for item in [c_skip, c_in, c_out, weightings]]
        return c_skip, c_in, c_out, c_noise, weightings
    
def append_dims(x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])



def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    return torch.linspace(t_max, t_min, n + 1).to(device)