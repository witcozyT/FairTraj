"""Density estimation and density-aware graph construction for FairTraj.

This module contains task-agnostic preprocessing code. It expects trajectory
files in the common RNTrajRec-style text format used by the original project:
one point per line as ``timestamp latitude longitude`` and a line beginning
with ``-`` as the trajectory separator.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from math import radians
from pathlib import Path
from typing import Iterable, List, Tuple, Union

from geopy.distance import geodesic
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from sklearn.neighbors import BallTree
from torch_geometric.data import Data
from tqdm import tqdm

from fairtraj.models.dwgat import eval as eval_dwgat
from fairtraj.models.dwgat import train as train_dwgat
from fairtraj.utils.quadtree import QuadTree
from fairtraj.utils.logger import get_logger
from fairtraj.utils.utils import divide_grids, resample_trajectory


PORTO_BOUNDS = {
    "lati_min": 41.111975,
    "long_min": -8.667057,
    "lati_max": 41.177462,
    "long_max": -8.585305,
}


@dataclass(frozen=True)
class DensityConfig:
    bounds: dict
    quadtree_capacity: int = 6000
    search_radius_m: float = 3000.0


def read_trajectories(path: Union[str, Path], with_time: bool = False) -> List[List[List[float]]]:
    trajectories = []
    points = []
    with Path(path).open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].startswith("-"):
                if len(points) > 1:
                    trajectories.append(points)
                points = []
                continue
            if with_time:
                points.append([float(parts[1]), float(parts[2]), float(parts[0])])
            else:
                points.append([float(parts[1]), float(parts[2])])
    if len(points) > 1:
        trajectories.append(points)
    return trajectories


def build_quadtree(bounds: dict, capacity: int) -> QuadTree:
    return QuadTree(
        [
            bounds["lati_min"],
            bounds["long_min"],
            bounds["lati_max"],
            bounds["long_max"],
        ],
        capacity=capacity,
    )


def restricted_kernel_density_estimation(trajectories: Iterable[Iterable[Iterable[float]]], config: DensityConfig) -> Tuple[QuadTree, list]:
    qdtree = build_quadtree(config.bounds, config.quadtree_capacity)
    for traj in trajectories:
        for lat, lon, *_ in traj:
            qdtree.insert(point=(lat * 1e6, lon * 1e6))

    nodes = qdtree.get_nodes()
    points_radians = np.array(
        [[radians(node.center.x / 1e6), radians(node.center.y / 1e6)] for node in nodes]
    )
    ball_tree = BallTree(points_radians, metric="haversine")

    search_lists = []
    for point in points_radians:
        ids = ball_tree.query_radius([point], r=config.search_radius_m / 6371000)[0]
        search_lists.append([nodes[idx] for idx in ids])

    densities = []
    for i, neighbors in enumerate(search_lists):
        if len(neighbors) > 1:
            kde_data = np.array([[node.center.x, node.center.y] for node in neighbors]).T
            kde = gaussian_kde(kde_data, bw_method="scott")
            densities.append(kde([[nodes[i].center.x], [nodes[i].center.y]])[0])
        else:
            densities.append(0.0)

    for item in zip(nodes, densities):
        qdtree.update_density(item)
    return qdtree, nodes


def mean_pool_density(traj: Iterable[Iterable[float]], qdtree: QuadTree) -> float:
    densities = []
    for lat, lon, *_ in traj:
        _, node = qdtree.find((lat * 1e6, lon * 1e6))
        if node is None:
            raise ValueError(f"Point ({lat}, {lon}) is outside the quadtree bounds.")
        densities.append(node.density)
    return float(np.mean(densities))


def find_grid_id(lat, lon, lat_grids, lon_grids):
    lat_index = np.searchsorted(lat_grids, lat, side="right") - 1
    lon_index = np.searchsorted(lon_grids, lon, side="right") - 1
    lat_index = int(np.clip(lat_index, 0, len(lat_grids) - 1))
    lon_index = int(np.clip(lon_index, 0, len(lon_grids) - 1))
    return lat_index * len(lat_grids) + lon_index


def calculate_trajectory_attributes(trajectory, density, bounds, grids_num=16):
    total_distance = 0.0
    lasting_time = trajectory[-1][2] - trajectory[0][2]
    points_num = len(trajectory)
    if lasting_time <= 0 or points_num <= 1:
        raise ValueError("Each trajectory must contain at least two time-ordered points.")

    departure_time = trajectory[0][2]
    time_slot = int((departure_time % 86400) // 300)

    for i in range(len(trajectory) - 1):
        point_a = (trajectory[i][0], trajectory[i][1])
        point_b = (trajectory[i + 1][0], trajectory[i + 1][1])
        total_distance += geodesic(point_a, point_b).meters

    avg_distance = total_distance / (points_num - 1)
    avg_speed = total_distance / lasting_time
    lat_grids, lon_grids = divide_grids(grids_num=grids_num, boundary=bounds)
    sid = find_grid_id(trajectory[0][0], trajectory[0][1], lat_grids, lon_grids)
    eid = find_grid_id(trajectory[-1][0], trajectory[-1][1], lat_grids, lon_grids)

    return [time_slot, total_distance, lasting_time, points_num, avg_distance, avg_speed, density, sid, eid]


def normalize_stats(values):
    return np.mean(values, axis=0), np.std(values, axis=0)


def diffusion_arrays(trajectories, qdtree, bounds, traj_length, density_min=None, density_max=None):
    densities = np.asarray([mean_pool_density(traj, qdtree) for traj in trajectories])
    if density_min is None:
        density_min = densities.min()
    if density_max is None:
        density_max = densities.max()
    if density_max > density_min:
        densities = (densities - density_min) / (density_max - density_min)

    attrs = np.asarray(
        [calculate_trajectory_attributes(traj, density, bounds) for traj, density in zip(trajectories, densities)],
        dtype=np.float32,
    )
    trajs = np.asarray(
        [resample_trajectory(np.asarray([[pt[0], pt[1]] for pt in traj], dtype=np.float32), traj_length) for traj in trajectories],
        dtype=np.float32,
    )
    return trajs, attrs, densities


def prepare_diffusion_dataset(
    trajectories,
    conditions,
    qdtree,
    output_dir,
    bounds,
    traj_length=60,
    target_quantile=20,
    pool_trajectories=None,
    pool_conditions=None,
):
    train_raw_densities = np.asarray([mean_pool_density(traj, qdtree) for traj in trajectories])
    pool_raw_densities = None
    if pool_trajectories is not None:
        pool_raw_densities = np.asarray([mean_pool_density(traj, qdtree) for traj in pool_trajectories])
    stats_densities = train_raw_densities if pool_raw_densities is None else np.concatenate([train_raw_densities, pool_raw_densities])
    density_min = stats_densities.min()
    density_max = stats_densities.max()
    trajs, attrs, densities = diffusion_arrays(
        trajectories,
        qdtree,
        bounds,
        traj_length,
        density_min=density_min,
        density_max=density_max,
    )

    pool_trajs = None
    pool_attrs = None
    if pool_trajectories is not None:
        pool_trajs, pool_attrs, _ = diffusion_arrays(
            pool_trajectories,
            qdtree,
            bounds,
            traj_length,
            density_min=density_min,
            density_max=density_max,
        )

    stats_attrs = attrs if pool_attrs is None else np.concatenate([attrs, pool_attrs], axis=0)
    stats_trajs = trajs if pool_trajs is None else np.concatenate([trajs, pool_trajs], axis=0)

    attrs_mean, attrs_std = normalize_stats(stats_attrs[:, 1:7])
    attrs_std = np.where(attrs_std == 0, 1.0, attrs_std)
    coords_mean, coords_std = normalize_stats(stats_trajs.reshape(-1, 2))
    coords_std = np.where(coords_std == 0, 1.0, coords_std)

    attrs_norm = attrs.copy()
    attrs_norm[:, 1:7] = (attrs_norm[:, 1:7] - attrs_mean) / attrs_std
    trajs_norm = (trajs - coords_mean) / coords_std
    if pool_attrs is not None:
        pool_attrs_norm = pool_attrs.copy()
        pool_attrs_norm[:, 1:7] = (pool_attrs_norm[:, 1:7] - attrs_mean) / attrs_std

    output_dir = Path(output_dir)
    with (output_dir / "trajs.pkl").open("wb") as f:
        pickle.dump(trajs_norm, f)
    with (output_dir / "attrs.pkl").open("wb") as f:
        pickle.dump(attrs_norm, f)
    with (output_dir / "stats.pkl").open("wb") as f:
        pickle.dump(
            {
                "attrs_mean": attrs_mean,
                "attrs_std": attrs_std,
                "coords_mean": coords_mean,
                "coords_std": coords_std,
            },
            f,
        )

    threshold = np.percentile(densities, target_quantile)
    source_ids = densities >= threshold
    target_ids = densities < threshold
    split_items = {
        "source_trajs_train.pkl": trajs_norm[source_ids],
        "source_attrs_train.pkl": attrs_norm[source_ids],
        "source_conditions_train.pkl": conditions[source_ids],
        "target_trajs_train.pkl": trajs_norm[target_ids],
        "target_attrs_train.pkl": attrs_norm[target_ids],
        "target_conditions_train.pkl": conditions[target_ids],
    }
    for name, value in split_items.items():
        with (output_dir / name).open("wb") as f:
            pickle.dump(value, f)

    if pool_attrs is not None and pool_conditions is not None:
        with (output_dir / "attrs_pool.pkl").open("wb") as f:
            pickle.dump(pool_attrs_norm, f)
        with (output_dir / "conditions_pool.pkl").open("wb") as f:
            pickle.dump(pool_conditions, f)


def build_density_graph(trajectories, qdtree: QuadTree, nodes: list) -> Data:
    node_to_index = {str(node): idx for idx, node in enumerate(nodes)}
    node_features = {}
    edges = set()

    for idx, node in enumerate(nodes):
        node_features[idx] = {
            "density": node.density,
            "lat": node.center.x / 1e6,
            "lon": node.center.y / 1e6,
        }
        edges.add((idx, idx))

    for traj in trajectories:
        node_seq = []
        for lat, lon, *_ in traj:
            _, node = qdtree.find([lat * 1e6, lon * 1e6])
            if node is None:
                raise ValueError(f"Point ({lat}, {lon}) is outside the quadtree bounds.")
            node_seq.append(node_to_index[str(node)])
        for i in range(1, len(node_seq)):
            if node_seq[i - 1] != node_seq[i]:
                edges.add((node_seq[i - 1], node_seq[i]))

    edge_index = torch.tensor(list(edges), dtype=torch.long).T
    features = pd.DataFrame.from_dict(node_features, orient="index")
    features = (features - features.mean()) / features.std()
    x = torch.tensor(features.values, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def density_aware_conditions(trajectories, qdtree: QuadTree, nodes: list, embeddings: np.ndarray) -> np.ndarray:
    node_to_embedding = {str(node): emb for node, emb in zip(nodes, embeddings)}
    conditions = []
    for traj in tqdm(trajectories, desc="Building density-aware conditions"):
        condition = []
        for lat, lon, *_ in traj:
            _, node = qdtree.find([lat * 1e6, lon * 1e6])
            if node is None:
                raise ValueError(f"Point ({lat}, {lon}) is outside the quadtree bounds.")
            condition.append(node_to_embedding[str(node)])
        conditions.append(condition)
    return np.asarray(conditions)


def run_core_preprocessing(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__, output_dir / "preprocessing.log")
    logger.info("Loading trajectories from %s", args.trajectories)
    trajectories = read_trajectories(args.trajectories, with_time=True)
    config = DensityConfig(bounds=PORTO_BOUNDS, quadtree_capacity=args.capacity, search_radius_m=args.search_radius)

    logger.info("Estimating density with %s trajectories", len(trajectories))
    qdtree, nodes = restricted_kernel_density_estimation(trajectories, config)
    logger.info("Constructing density-aware graph with %s quadtree nodes", len(nodes))
    graph_data = build_density_graph(trajectories, qdtree, nodes)
    logger.info("Training DW-GAT for %s epochs", args.epochs)
    model = train_dwgat(graph_data, save_path=output_dir / "dwgat.pth", epochs=args.epochs, logger=logger)
    embeddings = eval_dwgat(model, graph_data, save_path=output_dir / "dwgat_embeddings.png", save_fig=args.save_fig)
    logger.info("Building density-aware conditional signals")
    fixed_trajs = [
        resample_trajectory(np.asarray([[pt[0], pt[1]] for pt in traj], dtype=np.float32), args.traj_length)
        for traj in trajectories
    ]
    conditions = density_aware_conditions(fixed_trajs, qdtree, nodes, embeddings)

    pool_trajectories = None
    pool_conditions = None
    if args.pool_trajectories:
        logger.info("Loading held-out generation pool from %s", args.pool_trajectories)
        pool_trajectories = read_trajectories(args.pool_trajectories, with_time=True)
        fixed_pool_trajs = [
            resample_trajectory(np.asarray([[pt[0], pt[1]] for pt in traj], dtype=np.float32), args.traj_length)
            for traj in pool_trajectories
        ]
        logger.info("Building density-aware conditional signals for %s held-out trajectories", len(pool_trajectories))
        pool_conditions = density_aware_conditions(fixed_pool_trajs, qdtree, nodes, embeddings)

    with (output_dir / "qdtree.pkl").open("wb") as f:
        pickle.dump(qdtree, f)
    with (output_dir / "nodes.pkl").open("wb") as f:
        pickle.dump(nodes, f)
    torch.save(graph_data, output_dir / "graph_data.pt")
    with (output_dir / "density_conditions.pkl").open("wb") as f:
        pickle.dump(conditions, f)
    prepare_diffusion_dataset(
        trajectories,
        conditions,
        qdtree,
        output_dir,
        PORTO_BOUNDS,
        traj_length=args.traj_length,
        target_quantile=args.target_quantile,
        pool_trajectories=pool_trajectories,
        pool_conditions=pool_conditions,
    )
    logger.info("Saved preprocessing artifacts to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FairTraj density-aware graph representations.")
    parser.add_argument("--trajectories", required=True, help="Path to trajectory text file.")
    parser.add_argument("--output-dir", default="outputs/fairtraj_core", help="Directory for generated core artifacts.")
    parser.add_argument("--capacity", type=int, default=6000, help="Quadtree leaf capacity.")
    parser.add_argument("--search-radius", type=float, default=3000.0, help="KDE search radius in meters.")
    parser.add_argument("--epochs", type=int, default=3000, help="DWGAT training epochs.")
    parser.add_argument("--traj-length", type=int, default=60, help="Fixed trajectory length used by DDPM.")
    parser.add_argument("--target-quantile", type=float, default=20, help="Low-density percentile used as target domain.")
    parser.add_argument("--pool-trajectories", default=None, help="Optional held-out trajectory file used only for augmentation conditions.")
    parser.add_argument("--save-fig", action="store_true", help="Save a PCA visualization of DWGAT embeddings.")
    run_core_preprocessing(parser.parse_args())


if __name__ == "__main__":
    main()
