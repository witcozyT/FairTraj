# Data

Large trajectory datasets are not included in this repository.

Expected input for the core FairTraj preprocessing pipeline:

```text
timestamp latitude longitude
timestamp latitude longitude
-1
timestamp latitude longitude
...
-2
```

Place public or locally authorized datasets under this directory, for example:

```text
data/
  porto/
    trajectories.txt
```

Do not commit raw datasets, processed `.pkl` files, graph caches, or generated condition files unless redistribution rights are explicit.
