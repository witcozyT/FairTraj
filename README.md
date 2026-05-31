# FairTraj

FairTraj is a density-aware generative data augmentation method for fairness in downstream trajectory learning tasks.

![FairTraj overview](assets/fairtraj_overview.png)

## Project Structure

```text
.
├── assets/
│   └── fairtraj_overview.png
├── configs/
│   └── fairtraj_core.yaml
├── src/
│   └── fairtraj/
│       ├── models/
│       │   ├── ddpm.py
│       │   └── dwgat.py
│       ├── generation/
│       │   └── generate.py
│       ├── preprocessing/
│       │   └── density.py
│       ├── training/
│       │   └── train_ddpm.py
│       └── utils/
│           ├── ema.py
│           ├── quadtree.py
│           └── utils.py
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

## Requirements

You can install the required packages in your python environment by:

```text
Python == 3.8
torch
pandas
numpy
matplotlib
scipy
scikit-learn
torch-geometric
tqdm
geopy
PyYAML
```

Install the local package before running the commands below:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Format

The core preprocessing command expects a trajectory text file with one GPS point per line:

```text
timestamp latitude longitude
timestamp latitude longitude
-1
timestamp latitude longitude
...
-2
```

Lines beginning with `-` separate trajectories. Coordinates are expected as latitude and longitude in decimal degrees.

## Workflow

FairTraj contains three runnable stages.

### 1. Estimate Trajectory Density and Build Density-Aware Conditions

Given raw trajectories, estimate point/trajectory density, construct the density-aware graph, train DW-GAT, build density-aware conditional signals, and prepare DDPM tensors:

```bash
python -m fairtraj.preprocessing.density \
  --trajectories /path/to/trajectories.txt \
  --pool-trajectories /path/to/heldout_trajectories.txt \
  --output-dir outputs/fairtraj_core \
  --capacity 6000 \
  --search-radius 3000 \
  --epochs 3000
```

This produces:

- `qdtree.pkl`: fitted quadtree with node densities
- `nodes.pkl`: quadtree leaf nodes
- `graph_data.pt`: density-aware graph data
- `dwgat.pth`: DWGAT checkpoint
- `density_conditions.pkl`: trajectory-level density-aware condition sequences
- `trajs.pkl`, `attrs.pkl`, `stats.pkl`: normalized DDPM training data and normalization statistics
- `source_*_train.pkl`, `target_*_train.pkl`: density-based source/target splits for FairTraj training
- `attrs_pool.pkl`, `conditions_pool.pkl`: held-out original trajectories prepared as generation attributes and density-aware conditions
- `preprocessing.log`: preprocessing and DW-GAT training log

For augmentation, FairTraj uses a held-out pool from the original trajectories as generation conditions. This pool is provided through `--pool-trajectories` and is not part of the DDPM training split. When the pool is provided, `stats.pkl` is fitted over the training and held-out trajectories together, and the pool files are prepared with the same DW-GAT node embeddings and normalization statistics as the training tensors.

### 2. Train Density-Aware DDPM

Train the FairTraj denoising diffusion model with the prepared source/target tensors:

```bash
python -m fairtraj.training.train_ddpm \
  --config configs/fairtraj_core.yaml \
  --data-dir outputs/fairtraj_core \
  --output-dir outputs/ddpm
```

This produces DDPM checkpoints such as:

```text
outputs/ddpm/models/unet_200.pt
```

Training logs are saved to `outputs/ddpm/logs/train_ddpm.log`.

### 3. Generate Augmented Trajectories

Use a trained DDPM checkpoint and the held-out condition pool to synthesize augmented trajectories:

```bash
python -m fairtraj.generation.generate \
  --config configs/fairtraj_core.yaml \
  --stats outputs/fairtraj_core/stats.pkl \
  --attrs outputs/fairtraj_core/attrs_pool.pkl \
  --conditions outputs/fairtraj_core/conditions_pool.pkl \
  --checkpoint outputs/ddpm/models/unet_200.pt \
  --output-dir outputs/augmented \
  --sample-temperature 0.02
```

During generation, FairTraj samples held-out pool conditions by density with a `WeightedRandomSampler`. The default probability is proportional to `exp(-density / sample_temperature)`, so lower-density trajectories are selected more often for augmentation.

This saves:

```text
outputs/augmented/augmented_trajs.pkl
```

Generation logs are saved to `outputs/augmented/generate.log`.

## Configuration

The default core configuration is provided in `configs/fairtraj_core.yaml`. It documents the Porto spatial bounds used in the original experiments and the default DWGAT / diffusion hyperparameters.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{10.1145/3770855.3817841,
  author = {Wang, Tao and Yao, Yuanyuan and Wei, Yian and Zhu, Junhao and Alinejad Rokny, Hamid and Chen, Lu},
  title = {FairTraj: Density-Aware Generative Data Augmentation for Fairness in Downstream Trajectory Learning Tasks},
  year = {2026},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3770855.3817841},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)},
  location = {Jeju Island, Republic of Korea},
  numpages = {10}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
