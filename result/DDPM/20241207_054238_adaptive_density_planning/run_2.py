import argparse
import json
import time
import os.path as osp
import numpy as np
from tqdm.auto import tqdm
import npeet.entropy_estimators as ee
import pickle
import pathlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from ema_pytorch import EMA
import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_dim = self.dim // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim)).to(device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ff = nn.Linear(width, width)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return x + self.ff(self.act(x))
class UrbanPlanner(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256, hidden_layers=3):
        super().__init__()
        self.time_mlp = SinusoidalEmbedding(embedding_dim)
        self.input_mlp1 = SinusoidalEmbedding(embedding_dim, scale=25.0)
        self.input_mlp2 = SinusoidalEmbedding(embedding_dim, scale=25.0)

        # 修改输出维度以匹配数据
        self.global_branch = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            *[ResidualBlock(hidden_dim) for _ in range(hidden_layers)],
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )

        self.local_branch = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            *[ResidualBlock(hidden_dim) for _ in range(hidden_layers)],
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)
        )

        self.cost_branch = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            *[ResidualBlock(hidden_dim) for _ in range(hidden_layers)],
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)
        )

    def forward(self, x, t):
        # 处理输入维度
        x1 = x[:, :8].mean(dim=1)  # 将8维特征平均为1维
        x2 = x[:, 8:16].mean(dim=1)  # 将8维特征平均为1维

        # 生成嵌入
        x1_emb = self.input_mlp1(x1)  # [batch_size, embedding_dim]
        x2_emb = self.input_mlp2(x2)  # [batch_size, embedding_dim]
        t_emb = self.time_mlp(t)      # [batch_size, embedding_dim]

        # 合并特征
        emb = torch.cat([x1_emb, x2_emb, t_emb], dim=-1)  # [batch_size, embedding_dim * 3]

        # 通过三个分支
        global_features = self.global_branch(emb)
        local_features = self.local_branch(emb)
        cost_features = self.cost_branch(emb)

        # 合并输出
        return torch.cat([global_features, local_features, cost_features], dim=-1)

class NoiseScheduler():
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        elif beta_schedule == "quadratic":
            self.betas = (torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2).to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        self.sqrt_alphas_cumprod = (self.alphas_cumprod ** 0.5).to(device)
        self.sqrt_one_minus_alphas_cumprod = ((1 - self.alphas_cumprod) ** 0.5).to(device)
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod).to(device)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1).to(device)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = ((1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod))

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
        return s1 * x_start + s2 * x_noise

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        return pred_prev_sample + variance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_timesteps", type=int, default=100)
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--beta_schedule", type=str, default="quadratic")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="run_0")

    config = parser.parse_args()
    final_infos = {}
    all_results = {}

    pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    dataset = datasets.get_urban_dataset(n_samples=1000)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    model = UrbanPlanner(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_size,
        hidden_layers=config.hidden_layers,
    ).to(device)

    ema_model = EMA(model, beta=0.995, update_every=10).to(device)
    noise_scheduler = NoiseScheduler(num_timesteps=config.num_timesteps, beta_schedule=config.beta_schedule)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_train_steps)

    train_losses = []
    print("Training model...")

    model.train()
    global_step = 0
    progress_bar = tqdm(total=config.num_train_steps)
    progress_bar.set_description("Training")

    start_time = time.time()

    while global_step < config.num_train_steps:
        for batch in dataloader:
            if global_step >= config.num_train_steps:
                break

            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long().to(device)

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)

            global_loss = F.mse_loss(noise_pred[:, :6], noise[:, :6])
            local_loss = F.mse_loss(noise_pred[:, 6:11], noise[:, 6:11])
            cost_loss = F.mse_loss(noise_pred[:, 11:], noise[:, 11:])

            loss = global_loss + local_loss + 0.5 * cost_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            ema_model.update()
            scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            train_losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1

    progress_bar.close()
    end_time = time.time()
    training_time = end_time - start_time

    # 评估
    model.eval()
    eval_losses = []

    for batch in dataloader:
        batch = batch[0].to(device)
        noise = torch.randn(batch.shape).to(device)
        timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long().to(device)

        noisy = noise_scheduler.add_noise(batch, noise, timesteps)
        noise_pred = model(noisy, timesteps)

        global_loss = F.mse_loss(noise_pred[:, :8], noise[:, :8])
        local_loss = F.mse_loss(noise_pred[:, 8:16], noise[:, 8:16])
        cost_loss = F.mse_loss(noise_pred[:, 16:], noise[:, 16:])

        loss = global_loss + local_loss + 0.5 * cost_loss
        eval_losses.append(loss.detach().item())

    eval_loss = np.mean(eval_losses)

    final_infos["urban_planning"] = {
        "means": {
            "training_time": training_time,
            "eval_loss": eval_loss,
            "global_loss": global_loss.item(),
            "local_loss": local_loss.item(),
            "cost_loss": cost_loss.item(),
        }
    }

    all_results["urban_planning"] = {
        "train_losses": train_losses,
        "urban_features": noise_pred.cpu().detach().numpy(),
    }

    with open(osp.join(config.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(osp.join(config.out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)
