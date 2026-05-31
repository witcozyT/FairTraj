"""Train the FairTraj density-aware DDPM."""

import argparse
import datetime
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler, TensorDataset

from fairtraj.models.ddpm import Guide_UNet
from fairtraj.utils.ema import EMAHelper


class GradualDomainSampler(Sampler):
    def __init__(self, source_indices, target_indices, batch_size, batches_per_epoch, keep_source=True, seed=42, increment=4):
        self.source_indices = list(source_indices)
        self.target_indices = list(target_indices)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.keep_source = keep_source
        self.epoch = 0
        self.increment = increment
        self.rng = random.Random(seed)

    def _sample(self, indices, k):
        if k <= 0:
            return []
        if len(indices) >= k:
            return self.rng.sample(indices, k=k)
        return [self.rng.choice(indices) for _ in range(k)]

    def __iter__(self):
        if self.keep_source:
            from_target = min((self.epoch + 1) * self.increment, self.batch_size - self.increment)
        else:
            from_target = min((self.epoch + 1) * self.increment, self.batch_size)
        from_source = self.batch_size - from_target

        for _ in range(self.batches_per_epoch):
            batch_indices = self._sample(self.target_indices, from_target) + self._sample(self.source_indices, from_source)
            self.rng.shuffle(batch_indices)
            yield batch_indices

        self.epoch += 1

    def __len__(self):
        return self.batches_per_epoch


def namespace_from_dict(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace_from_dict(v) for k, v in value.items()})
    return value


def load_config(path):
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f)
    model = raw.get("model", {})
    diffusion = raw.get("diffusion", {})
    data = raw.get("dataset", {})
    training = raw.get("training", {})
    config = {
        "data": {
            "dataset": data.get("name", "porto"),
            "traj_length": data.get("traj_length", 60),
        },
        "model": {
            "in_channels": model.get("in_channels", diffusion.get("in_channels", 2)),
            "out_ch": model.get("out_ch", diffusion.get("out_channels", 2)),
            "ch": model.get("ch", diffusion.get("ch", 128)),
            "cond_ch": model.get("cond_ch", diffusion.get("cond_ch", 128)),
            "ch_mult": model.get("ch_mult", diffusion.get("ch_mult", [1, 2, 2, 2])),
            "num_res_blocks": model.get("num_res_blocks", diffusion.get("num_res_blocks", 2)),
            "dropout": model.get("dropout", diffusion.get("dropout", 0.1)),
            "ema": model.get("ema", True),
            "ema_rate": model.get("ema_rate", 0.9999),
            "resamp_with_conv": model.get("resamp_with_conv", True),
        },
        "diffusion": {
            "beta_start": diffusion.get("beta_start", 0.0001),
            "beta_end": diffusion.get("beta_end", 0.05),
            "num_diffusion_timesteps": diffusion.get("num_diffusion_timesteps", 500),
        },
        "training": {
            "batch_size": training.get("batch_size", 256),
            "batches_per_epoch": training.get("batches_per_epoch", 200),
            "n_epochs": training.get("n_epochs", 200),
            "learning_rate": training.get("learning_rate", 3e-4),
            "sample_weight_temperature": training.get("sample_weight_temperature", 0.3),
        },
    }
    return namespace_from_dict(config)


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def gather(consts, t):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def q_xt_x0(x0, t, alpha_bar):
    mean = gather(alpha_bar, t) ** 0.5 * x0
    var = 1 - gather(alpha_bar, t)
    eps = torch.randn_like(x0).to(x0.device)
    return mean + (var ** 0.5) * eps, eps


def train(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    result_dir = output_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    unet = Guide_UNet(config).to(device)

    source_traj = torch.from_numpy(np.asarray(load_pickle(data_dir / "source_trajs_train.pkl"))).float()
    source_attr = torch.from_numpy(np.asarray(load_pickle(data_dir / "source_attrs_train.pkl"))).float()
    source_cond = torch.from_numpy(np.asarray(load_pickle(data_dir / "source_conditions_train.pkl"))).float()
    target_traj = torch.from_numpy(np.asarray(load_pickle(data_dir / "target_trajs_train.pkl"))).float()
    target_attr = torch.from_numpy(np.asarray(load_pickle(data_dir / "target_attrs_train.pkl"))).float()
    target_cond = torch.from_numpy(np.asarray(load_pickle(data_dir / "target_conditions_train.pkl"))).float()

    traj = torch.cat([source_traj, target_traj], dim=0)
    attrs = torch.cat([source_attr, target_attr], dim=0)
    conditions = torch.cat([source_cond, target_cond], dim=0)

    dataset = TensorDataset(traj, attrs, conditions)
    source_ids = torch.arange(0, source_traj.shape[0]).tolist()
    target_ids = torch.arange(source_traj.shape[0], traj.shape[0]).tolist()
    sampler = GradualDomainSampler(
        source_ids,
        target_ids,
        config.training.batch_size,
        config.training.batches_per_epoch,
        increment=args.target_increment,
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps, device=device)
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.training.learning_rate)

    ema_helper = EMAHelper(mu=config.model.ema_rate) if config.model.ema else None
    if ema_helper is not None:
        ema_helper.register(unet)

    stats = load_pickle(data_dir / "stats.pkl")
    density_mean = stats["attrs_mean"][-1]
    density_std = stats["attrs_std"][-1]
    losses = []

    for epoch in range(1, config.training.n_epochs + 1):
        epoch_losses = []
        unet.train()
        for train_x, attr, den_cond in dataloader:
            x0 = train_x.to(device).transpose(-1, -2)
            attr = attr.to(device)
            den_cond = den_cond.to(device).transpose(-1, -2)
            t = torch.randint(low=0, high=n_steps, size=(len(x0) // 2 + 1,), device=device)
            t = torch.cat([t, n_steps - t - 1], dim=0)[: len(x0)]

            xt, noise = q_xt_x0(x0, t, alpha_bar)
            pred_noise = unet(xt.float(), t, attr, den_cond)
            den_prob = attr[:, 6] * density_std + density_mean
            sample_weight = 1 + torch.exp(-den_prob / config.training.sample_weight_temperature)
            loss = F.mse_loss(noise.float(), pred_noise, reduction="none").mean(dim=[1, 2]) * sample_weight
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema_helper is not None:
                ema_helper.update(unet)
            epoch_losses.append(loss.item())

        losses.append(float(np.mean(epoch_losses)))
        print("Epoch {} | loss {:.6f} | {}".format(epoch, losses[-1], datetime.datetime.now().isoformat(timespec="seconds")))
        if epoch % args.save_every == 0 or epoch == config.training.n_epochs:
            torch.save(unet.state_dict(), model_dir / "unet_{}.pt".format(epoch))

    if ema_helper is not None:
        torch.save(ema_helper.state_dict(), model_dir / "ema.pt")
    np.savetxt(result_dir / "loss.csv", np.asarray(losses), delimiter=",")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(result_dir / "loss.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train FairTraj density-aware DDPM.")
    parser.add_argument("--config", default="configs/fairtraj_core.yaml")
    parser.add_argument("--data-dir", default="outputs/fairtraj_core")
    parser.add_argument("--output-dir", default="outputs/ddpm")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--target-increment", type=int, default=4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
