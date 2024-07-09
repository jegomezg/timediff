import numpy as np
import torch
import torch.nn as nn
from . import diff_models


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]


        config_diff = config["diffusion"]

        self.diffmodel = diff_models.diff_CSDI(config_diff)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


    def calc_loss_valid(self, observed_data, cond_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, cond_info, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps


    def calc_loss(
        self, observed_data, cond_info, is_train, set_t=-1
    ):
        B, L, K = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        predicted = self.diffmodel(noisy_data,cond_info, t)  # (B,K,L)
        predicted = predicted.permute(0,2,1)

        residual = (noise - predicted)
        num_eval = B * K * L
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss


    def generate(self, cond_info, observed_data, n_samples):

        B, L, K = observed_data.shape

        generated_samples = torch.zeros(B, n_samples, L, K)

        for i in range(n_samples):
            current_sample = torch.randn(B, L, K).to(self.device)

            for t in range(self.num_steps - 1, -1, -1):
                
                predicted = self.diffmodel(current_sample, cond_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted.permute(0,2,1))

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            print(i)

            generated_samples[:, i] = current_sample.detach()

        return generated_samples , observed_data

    def forward(self, batch, is_train=1):
        (
            observed_data,
            cond_info,

        ) = self.process_data(batch)


        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_info, is_train)

    def evaluate(self, batch, n_samples):
        
        (
            observed_data,
            cond_info,

        ) = self.process_data(batch)

        with torch.no_grad():

            samples = self.generate(cond_info, observed_data, n_samples)

        return samples


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        

        observed_data = torch.stack(tuple(batch["response"].values()), dim=1).to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
   
        if batch["cat_condition"]:
            cat_cond_info = torch.stack(tuple(batch["cat_condition"].values()), dim=1).to(self.device).float()
            cat_cond_info = cat_cond_info.permute(0, 2, 1)
        else:
            cat_cond_info = None

        if batch["cont_condition"]:
            cont_condition = torch.stack(tuple(batch["cont_condition"].values()), dim=1).to(self.device).float()
            cont_condition = cont_condition.permute(0, 2, 1)
        else:
            cont_condition = None

        
        cond_info = {'cat':cat_cond_info,
                     'cont':cont_condition}

        return (
            observed_data,
            cond_info
        )

