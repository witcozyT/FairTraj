"""Generate augmented trajectories with a trained FairTraj DDPM."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fairtraj.models.ddpm import Guide_UNet
from fairtraj.training.train_ddpm import load_config
from fairtraj.utils.utils import p_xt, resample_trajectory
from fairtraj.utils.logger import get_logger


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def generate(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__, output_dir / "generate.log")
    config = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Generating trajectories on device %s", device)

    attrs = torch.from_numpy(np.asarray(load_pickle(args.attrs))).float()
    conditions = torch.from_numpy(np.asarray(load_pickle(args.conditions))).float()
    dataset = TensorDataset(attrs, conditions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    stats = load_pickle(data_dir / "stats.pkl")
    coord_mean = stats["coords_mean"]
    coord_std = stats["coords_std"]
    len_mean = stats["attrs_mean"][2]
    len_std = stats["attrs_std"][2]

    model = Guide_UNet(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    n_steps = config.diffusion.num_diffusion_timesteps
    timesteps = args.timesteps
    skip = max(n_steps // timesteps, 1)
    seq = list(range(0, n_steps, skip))
    seq_next = [-1] + seq[:-1]
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps, device=device)

    generated = []
    with torch.no_grad():
        for attr, cond in dataloader:
            lengths = (attr[:, 3] * len_std + len_mean).int().clamp(min=2)
            attr = attr.to(device)
            cond = cond.to(device).transpose(-1, -2)
            x = torch.randn(attr.shape[0], 2, config.data.traj_length, device=device)
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.ones(x.shape[0], device=device) * i
                next_t = torch.ones(x.shape[0], device=device) * j
                pred_noise = model(x, t, attr, cond)
                x = p_xt(x, pred_noise, t, next_t, beta, args.eta)

            trajs = x.cpu().numpy()[:, :2, :]
            for idx in range(trajs.shape[0]):
                traj = resample_trajectory(trajs[idx].T, int(lengths[idx]))
                generated.append(traj * coord_std + coord_mean)

    output_path = output_dir / args.output_name
    with output_path.open("wb") as f:
        pickle.dump(generated, f)
    logger.info("Saved %s generated trajectories to %s", len(generated), output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate FairTraj augmented trajectories.")
    parser.add_argument("--config", default="configs/fairtraj_core.yaml")
    parser.add_argument("--data-dir", default="outputs/fairtraj_core")
    parser.add_argument("--attrs", default="outputs/fairtraj_core/attrs_pool.pkl")
    parser.add_argument("--conditions", default="outputs/fairtraj_core/conditions_pool.pkl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/augmented")
    parser.add_argument("--output-name", default="augmented_trajs.pkl")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
