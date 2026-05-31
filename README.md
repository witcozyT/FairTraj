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
│       ├── preprocessing/
│       │   └── density.py
│       └── utils/
│           ├── quadtree.py
│           └── trajectory.py
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

## Requirements

You can install the required packages in your python environment by:

```text
Python == 3.8
torch>=1.7
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

## Build FairTraj Core Artifacts

After preparing a trajectory text file, run:

```bash
python -m fairtraj.preprocessing.density \
  --trajectories /path/to/trajectories.txt \
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
