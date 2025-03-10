import sys
sys.path.append('/home/azureuser/Projects/TwoPSGen')

from CUBEPlot import plot_cube

from pathlib import Path
from typing import List, Optional, Type, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import nn, optim
from torchmetrics import MeanMetric

import traceback

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from einops import rearrange, reduce

from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt


class Diffusion(LightningModule):
    def __init__(
        self,
        model,
        *,
        loss_fn: nn.Module = nn.MSELoss(),
        learning_rate: float = 1e-4,
        data_std: float = 0.5,
        time_min: float = 0.002,
        time_max: float = 80.0,
        bins_max: int = 25,
        bins_rho: float = 7,
        ema_decay: float = 0.9,
        optimizer_type: Type[optim.Optimizer] = optim.AdamW,
        samples_path: str = "samples/",
        save_samples_every_n_epoch: int = 1,
        num_samples: int = 1,
        sample_steps: int = 1,
        use_ema: bool = True,
        sample_seed: int = 0,
        epochs_max = 100,
        **kwargs,
    ) -> None:
        super().__init__()
        self.max_epochs = epochs_max
        self.model = model
        self.model_ema = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        self.image_size = model.sample_size
        self.channels = model.in_channels
        self.dims = self.model.dims

        self.model_ema.requires_grad_(False)

        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate

        self.data_std = data_std
        self.time_min = time_min
        self.time_max = time_max
        self.bins_max = bins_max
        self.bins_rho = bins_rho

        self._loss_tracker = MeanMetric()

        Path(samples_path).mkdir(exist_ok=True, parents=True)

        self.samples_path = samples_path
        self.save_samples_every_n_epoch = save_samples_every_n_epoch
        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.use_ema = use_ema
        self.sample_seed = sample_seed
        
    def forward(
        self,
        images: torch.Tensor,
        times: torch.Tensor,
        ):

        if self.use_ema:
            model = self.model_ema
        else:
            pass #model = self.model

        skip_coef, out_coef, in_coef, noise_coef, _ = self.calc_coeffs(times)

        out = model(
            self.image_time_product(images,in_coef), 
            noise_coef
            )

        out = self.image_time_product(
            images,
            skip_coef,
        ) + self.image_time_product(
            out.sample,
            out_coef,
        )

        return out
    
    def training_step(self, image_sets, *args, **kwargs):
        loss = torch.zeros(1,device=self.device)

        for images in image_sets:
            if isinstance(images, list): 
                images = images[0]
            
            if self.global_step == 0:
                print(images.shape)
            
            p_mean, p_std = (-1.2,1.2)
            times = torch.exp(torch.randn((images.shape[0],),device=self.device)*p_std + p_mean)
            noise = torch.randn(images.shape, device=images.device)

            noise_image = images + self.image_time_product(
                noise,
                times,
            )

            skip_coef, out_coef, in_coef, noise_coef, loss_coef = self.calc_coeffs(times)

            out = self.model(
                self.image_time_product(noise_image,in_coef), 
                noise_coef
                )
                
            target = self.image_time_product(
                (images - self.image_time_product(noise_image,skip_coef)),
                (1/out_coef))

            eff_weight = loss_coef * out_coef.pow(2)

            if self.dims == 2:
                loss += torch.mean(
                    eff_weight*
                    ((out.sample-target)**2).mean(1).mean(1).mean(1)
                    )
            elif self.dims == 3:
                loss += torch.mean(
                    eff_weight*
                    ((out.sample-target)**2).mean(1).mean(1).mean(1).mean(1)
                    )
            else:
                raise Exception("Only 2D or 3D volumes supported for now!")
            
            if torch.isnan(loss):
                self.nan += 1
                print(f"Nan! {self.nan}'s Total.")
                return None


        self._loss_tracker(loss)

        self.log(
            "loss",
            self._loss_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar = True
        )
        

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_type(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.estimated_stepping_batches, eta_min = self.learning_rate/100)
        return [optimizer], [{'scheduler':scheduler,'name': 'learning_rate','interval':'step','frequency': 1}]

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.model_ema.update_parameters(self.model)
     
    def calc_coeffs(self, times: torch.Tensor):
        skip_coef = self.data_std**2 / (
            (times)**2 + self.data_std**2
        )
        out_coef = self.data_std * times / (times.pow(2) + self.data_std**2).pow(0.5)

        in_coef = 1/(times.pow(2) + self.data_std**2).pow(0.5)

        noise_coef = .25*torch.log(times)

        loss_coef = (times.pow(2) + self.data_std**2) / (times*self.data_std)**2

        return skip_coef, out_coef, in_coef, noise_coef, loss_coef

    def timesteps_to_times(self, timesteps: torch.LongTensor, bins: int):
        return (
            (
                self.time_min ** (1 / self.bins_rho)
                + timesteps
                / (bins - 1)
                * (
                    self.time_max ** (1 / self.bins_rho)
                    - self.time_min ** (1 / self.bins_rho)
                )
            )
            .pow(self.bins_rho)
            .clamp(0, self.time_max)
        )

    @rank_zero_only
    def on_train_start(self) -> None:
        pass
        '''
        self.save_samples(
            f"{0:05}",
            num_samples=self.num_samples,
            steps=self.bins_max,
            generator=torch.Generator(device=self.device).manual_seed(self.sample_seed),
        )
        '''
    @rank_zero_only
    def on_train_epoch_end(self) -> None:
        if (
            ((self.trainer.current_epoch + 1) % self.save_samples_every_n_epoch == 0)
            or self.trainer.current_epoch == (self.trainer.max_epochs - 1)
        ):
            self.save_samples(
                f"{(self.current_epoch+1):05}",
                num_samples=self.num_samples,
                steps=self.bins_max,
                generator=torch.Generator(device=self.device).manual_seed(
                    self.sample_seed
                ),
            )
    
    @torch.no_grad()
    def sample( 
        self,
        num_samples: int = 1,
        steps = None ,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        images = None,
        skip = None,
        cond_fn = None,
    ) -> torch.Tensor:
        
        if steps == None:
            steps = self.bins_max
        
        timesteps = torch.arange(0, steps ,device=self.device).long()
        times = self.timesteps_to_times(timesteps, steps)[None,:]

        if skip != None:
            timesteps = timesteps[:-skip]

        if images == None:
            shape = tuple([1, self.channels] + [self.image_size]*self.dims) #set this to 3 if doing 2d to 3d
            images = randn_tensor(shape, generator=generator, device=self.device) * times[:,-1]

        else:
            shape = tuple(images.shape)
            images = images + randn_tensor(shape, generator=generator, device=self.device) * times[:,timesteps[-1]]

            if num_samples > 1:
                print("multiple samples will not work for diffusion with an initalization, cause ram shit. idk try and change it if you want. tbh it's not that hard, im just lazy")

        sampler = self.sample_karras

        if num_samples == 1:
            return sampler( 
                generator = generator,
                timesteps=timesteps,
                times=times,
                shape=shape,
                images=images,
                cond_fn = cond_fn)
        else:
            out = []
            for i in range(num_samples):
                images = randn_tensor(shape, generator=generator, device=self.device) * times[:,-1]
                out.append(sampler(
                    generator = generator,
                    timesteps=timesteps,
                    times=times,
                    shape=shape,
                    images=images,
                    cond_fn = cond_fn
                ).cpu())
            images = torch.cat(out,axis=0)            
        return images
        
    @torch.no_grad()
    def sample_euler( 
        self,
        timesteps= None,
        times = None,
        shape = None,
        images = None,
        generator = None,
        cond_fn = None
    ) -> torch.Tensor:
                
        for i in reversed(timesteps[1:]):
            h = times[:,i-1]-times[:,i]
            d = (images-self.forward(images,times[:,i]))/times[:,i]
            images = images + h*d

            if cond_fn is not None:
                images = cond_fn(images)

        images = images.clamp(-1,1) 

        return images
    
    @torch.no_grad()
    def sample_heun( 
        self,
        timesteps= None,
        times = None,
        shape = None,
        images = None,
        generator = None,
        cond_fn = None
    ) -> torch.Tensor:
        
        alpha = 1
        for i in reversed(timesteps[1:]):
            h = times[:,i-1]-times[:,i]
            d = (images-self.forward(images,times[:,i]))/times[:,i]

            xp = images + alpha*h*d
            tp = times[:,i] + alpha*h

            if tp > self.time_min:
                dp = (xp - self.forward(xp,tp))/tp
                images = images + h*((1-1/(2*alpha))*d + 1/(2*alpha) * dp)
            else:
                images = images + h*d

            if cond_fn is not None:
                images = cond_fn(images)

        images = images.clamp(-1,1) 

        return images
    
    @torch.no_grad()
    def sample_karras( 
        self,
        generator= None,
        timesteps= None,
        times = None,
        shape = None,
        images = None,
        cond_fn = None
    ) -> torch.Tensor:
                

        s_min, s_max, s_noise, s_churn = (.05, 50, 1.003, 40)  

        for i in reversed(timesteps[1:]):
                
            e = randn_tensor(shape, generator=generator, device=self.device) * s_noise
            
            if times[:,i].item() > s_min and times[:,i].item() < s_max:
                gamma = torch.tensor([s_churn/self.bins_max,(2**.5)-1]).min()
            else:
                gamma = 0

            th = times[:,i] + gamma*times[:,i]
            xh = images + torch.sqrt(th**2 - times[:,i]**2) * e
            d = (xh-self.forward(xh,th))/th

            images = xh + (times[:,i-1] - th)*d

            if times[:,i-1].item() > self.time_min:
                dp = (images-self.forward(images,times[:,i-1]))/times[:,i-1]
                images = xh + (times[:,i-1] - th)*(.5*d + .5*dp)

            if cond_fn is not None:
                images = cond_fn(images)

        images = images.clamp(-1,1) 

        return images  
    
    @torch.no_grad()
    def save_samples(
        self,
        filename: str,
        num_samples: int = 1,
        steps: int = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        images = None,
        skip = None,
        save_numpy = False
    ):
        try: 
            samples = self.sample(
                num_samples=num_samples,
                steps=steps,
                generator=generator,
                images = images,
                skip = skip, 
            )
        except:
            #print long stack traces while debugging
            with open("traceback.txt","w") as f:
                traceback.print_exc(file=f)
            print("Saved tracback in traceback.txt!")
            exit(1)

        
        samples_dim = len(samples.shape)-2
        samples = samples.cpu().detach().numpy()
        samples = samples*0.5 + 0.5
        
        if save_numpy:
            np.save(f"{self.samples_path}/{filename}.npy", samples)

        for i in range(num_samples):
            if samples_dim == 3:
                plot_s = np.moveaxis(samples[i],0,-1)
                plot_cube(plot_s, f"{self.samples_path}/{filename}_{i}.png")
            if samples_dim == 2:
                self.plot_slice(samples[i], f"{self.samples_path}/{filename}_{i}.png")

        del samples
        torch.cuda.empty_cache()

    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b..., b -> b...", images, times)

    @staticmethod
    def plot_slice(im,savedir):

        im = np.moveaxis(im,0,-1)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(im)
        ax.axis("off")
        fig.tight_layout()    
        plt.savefig(savedir,transparent=True,dpi=300)
        plt.close()

    