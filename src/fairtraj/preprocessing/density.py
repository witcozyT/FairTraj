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
from typing import Iterable

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


def read_trajectories(path: str | Path, with_time: bool = False) -> list[list[list[float]]]:
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


def restricted_kernel_density_estimation(
    trajectories: Iterable[Iterable[Iterable[float]]],
    config: DensityConfig,
) -> tuple[QuadTree, list]:
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
    trajectories = read_trajectories(args.trajectories)
    config = DensityConfig(bounds=PORTO_BOUNDS, quadtree_capacity=args.capacity, search_radius_m=args.search_radius)

    qdtree, nodes = restricted_kernel_density_estimation(trajectories, config)
    graph_data = build_density_graph(trajectories, qdtree, nodes)
    model = train_dwgat(graph_data, save_path=output_dir / "dwgat.pth", epochs=args.epochs)
    embeddings = eval_dwgat(model, graph_data, save_path=output_dir / "dwgat_embeddings.png", save_fig=args.save_fig)
    conditions = density_aware_conditions(trajectories, qdtree, nodes, embeddings)

    with (output_dir / "qdtree.pkl").open("wb") as f:
        pickle.dump(qdtree, f)
    with (output_dir / "nodes.pkl").open("wb") as f:
        pickle.dump(nodes, f)
    torch.save(graph_data, output_dir / "graph_data.pt")
    with (output_dir / "density_conditions.pkl").open("wb") as f:
        pickle.dump(conditions, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FairTraj density-aware graph representations.")
    parser.add_argument("--trajectories", required=True, help="Path to trajectory text file.")
    parser.add_argument("--output-dir", default="outputs/fairtraj_core", help="Directory for generated core artifacts.")
    parser.add_argument("--capacity", type=int, default=6000, help="Quadtree leaf capacity.")
    parser.add_argument("--search-radius", type=float, default=3000.0, help="KDE search radius in meters.")
    parser.add_argument("--epochs", type=int, default=3000, help="DWGAT training epochs.")
    parser.add_argument("--save-fig", action="store_true", help="Save a PCA visualization of DWGAT embeddings.")
    run_core_preprocessing(parser.parse_args())


if __name__ == "__main__":
    main()
