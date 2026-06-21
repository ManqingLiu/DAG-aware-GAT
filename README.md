# DAG-aware GAT for Causal Effect Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)

## Overview

This repository contains the implementation of a **DAG-aware Graph Attention Network (GAT)** for causal effect
estimation, as described in our paper *DAG-aware GAT for Causal Effect Estimation* (accepted at
[*Transactions on Machine Learning Research*](https://openreview.net/); arXiv preprint:
[2410.10044](https://arxiv.org/abs/2410.10044)).

Our model embeds the causal DAG as a **hard structural inductive bias** directly into the multi-head attention
mechanism: attention scores between variables that are not causally connected are masked to negative infinity,
so each node's representation is strictly a function of its causal ancestors. In contrast to standard Transformers,
the GAT encoder **omits Layer Normalization** to preserve the heterogeneous scales of causal variables (e.g.,
binary treatment indicators vs. continuous outcomes), which we find is critical for unbiased propensity score and
bridge-function estimation.

## Key Features

- DAG-constrained multi-head attention (DAG as a hard inductive bias on information flow)
- LayerNorm-free GAT encoder, preserving heterogeneous variable scales
- Support for multiple causal inference methods: G-formula, IPW, and AIPW
- Joint or separate training of propensity score and outcome models for AIPW
- Proximal causal inference via the Neural Maximum Moment Restriction (NMMR) framework, with both
  U-statistic and V-statistic variants
- Baselines included for comparison: GRF, MLP, standard GNN, Transformer (with LayerNorm), and a standard
  fully-connected GAT (no DAG mask)
- DAG misspecification ablations for proximal inference (reversed treatment-outcome edge, missing proxy edge)

## Project Structure

Our project is organized as follows:
```
.
├── README.md
├── config
│   ├── dag
│   └── train
├── data
│   ├── acic
│   └── lalonde
├── experiments
│   ├── results
│   └── tuning
├── requirements.txt
├── scripts
│   ├── myjob.sh
│   └── myjob_proximal.sh
├── src
│   ├── data
│   ├── dataset.py
│   ├── evaluate
│   ├── experiment.py
│   ├── experiment_proximal.py
│   ├── models
│   ├── train
│   ├── utils.py
│   ├── utils_proximal
│   └── visualization
└── tests
```

- `config/`: Contains configuration files for DAG structures and training parameters.
- `data/`: Contains data loading and preprocessing scripts.
- `experiments/`: Holds experimental results.
- `scripts/`: Contains scripts for running the experiments.
- `src/`: The main source code directory.
  - `data/`: Data loading and preprocessing modules.
  - `evaluate/`: Evaluation metrics and functions.
  - `models/`: DAG-aware GAT model, baseline architectures, and their loss functions. Notable files:
    - `dag_transformer_integration.py`: training/prediction interface for the DAG-aware GAT encoder.
    - `gat_baseline.py` / `gat_integration.py`: standard fully-connected GAT baseline (no DAG mask).
    - `NMMR/`: proximal-inference estimator (Neural Maximum Moment Restriction; U- and V-statistic variants).
  - `train/`: Programs to compute pseudo ATE/CATE (see descriptions in Hyper-parameter tuning section in our paper) 
  and the computed values. 
  - `utils/`: Utility functions for data processing and model training.
  - `utils_proximal/`: Utility functions for proximal inference.
  - `visualization/`: Code for generating plots and visualizations.
  - `experiment.py`: Main script for running experiments.
  - `experiment_proximal.py`: Main script for running proximal inference experiments.
- `tests/`: Unit tests for the project.


## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Datasets

We evaluate our model on four datasets:

1. Lalonde-CPS
2. Lalonde-PSID
3. ACIC
4. Demand dataset (for proximal inference)

Data preprocessing scripts and instructions can be found in the `data/` directory.

## Experiments

### Lalonde-CPS, Lalonde-PSID and ACIC

To reproduce the experiments for Lalonde-CPS, Lalonde-PSID and ACIC, run:

```bash
python3 src/experiment.py \
        --config config/train/<DATA_NAME>/<DATA_NAME>_sample<SAMPLE_ID>.json \
        --dag <DAG_TYPE> \
        --estimator <ESTIMATOR_TYPE> \
        --data_name <DATA_NAME>
```

#### Parameters

- **CONFIG_FILE**: The configuration file for the experiment
  - Location: `config/train/<DATA_NAME>/`
  - Naming Convention: `<DATA_NAME>_sample<SAMPLE_ID>.json`
  - Examples:
    - `acic_sample1.json`
    - `lalonde_cps_sample2.json`
    - `lalonde_psid_sample3.json`

- **DAG_TYPE**: The type of Directed Acyclic Graph (DAG) to use
  - Options:
    - `dag_g_formula`
    - `dag_ipw`
    - `dag_aipw`

- **ESTIMATOR_TYPE**: The type of estimator to use
  - Options:
    - `g-formula`
    - `ipw`
    - `aipw`

- **DATA_NAME**: The name of the dataset
  - Options:
    - `lalonde_cps`
    - `lalonde_psid`
    - `acic`

- **SAMPLE_ID**: The sample ID for the experiment
  - A numeric value from 1 to 10 (e.g., 1, 2, 3, ...)

#### Example

```bash
python3 src/experiment.py \
        --config config/train/lalonde_cps/lalonde_cps_sample1.json \
        --dag dag_ipw \
        --estimator ipw \
        --data_name lalonde-cps
```

#### Note for AIPW
To get the result where you train outcome regression and propensity score models separately, 
you can run the following command:

1. Get predictions for outcome regression (e.g. for ACIC):
```bash
python3 src/experiment.py \
        --config config/train/acic/acic_sample1.json \
        --dag dag_g_formula \
        --estimator g-formula \
        --data_name acic
```

2. Get predictions for propensity score (e.g. for ACIC):
```bash
python3 src/experiment.py \
        --config config/train/acic/acic_sample1.json \
        --dag dag_ipw \
        --estimator ipw \
        --data_name acic
```

3. Plug in the predicted values to AIPW estimator (e.g. for ACIC):
```bash
python3 src/evaluate/acic/evaluate_metrics.py \
        --data_name acic \
        --estimator aipw \
        --sample_id 1

````

### Demand Dataset 

```bash
python3 src/experiment_proximal.py \
        --dag <DAG_CONFIG_FILE> \
        --config config/train/proximal/nmmr_<STATISTICS>_z_transformer_n<SAMPLE_SIZE>.json \
        --results_dir <RESULTS_DIRECTORY> \
        --sample_index <SAMPLE_INDEX>
```

#### Parameters

- **DAG_CONFIG_FILE**: The configuration file for the Directed Acyclic Graph (DAG)
  - Location: `config/dag/`
  - Examples:
    - `proximal_dag_z.json` — correctly specified DAG.
    - `proximal_dag_misspecified.json` — reversed treatment/outcome edge ($Y \to A$ instead of $A \to Y$);
      used for the structural-misspecification ablation in the paper.
    - `proximal_dag_misspecified2.json` — outcome proxy edge $W \to Y$ removed, breaking a key proximal
      identification condition.

- **STATISTICS**: The type of statistics used in the NMMR objective (see Eq. NMMR-U / NMMR-V in the paper)
  - Options: `u` (U-statistic; zeros out the kernel-matrix diagonal to avoid self-correlation)
  - or `v` (V-statistic; includes the diagonal; often more numerically stable)

- **SAMPLE_SIZE**: The size of the sample used in the experiment
  - Supported values: `1000`, `5000`, `10000`, `50000` (the camera-ready paper reports `1000`, `5000`, `10000`)

- **RESULTS_DIRECTORY**: The directory where results will be stored
  - Default: `experiments/results/proximal`

- **SAMPLE_INDEX**: The index of the sample to use for the experiment (form 0 to 19)
  - Example values: `0`, `1`, `2`, etc.

#### Example

```bash
python3 src/experiment_proximal.py \
        --dag config/dag/proximal_dag_z.json \
        --config config/train/proximal/nmmr_v_z_transformer_n50000.json \
        --results_dir experiments/results/proximal \
        --sample_index 1
```
You can also run the experiment using the provided script `scripts/myjob.sh` for lalonde-cps, lalonde-acic and ACIC; and
`scripts/myjob_proximal.sh` for demand by modifying the parameters in the script.

## Citation

Our paper *DAG-aware GAT for Causal Effect Estimation* has been accepted at **Transactions on Machine Learning
Research (TMLR)**. The official TMLR/OpenReview citation will be added here once the camera-ready DOI is issued.
In the meantime, please cite the arXiv preprint:

```bibtex
@article{liu2024dagaware,
      title={DAG-aware GAT for Causal Effect Estimation},
      author={Manqing Liu and David R. Bellamy and Andrew L. Beam},
      journal={arXiv preprint arXiv:2410.10044},
      year={2024},
      url={https://arxiv.org/abs/2410.10044},
      note={Accepted at Transactions on Machine Learning Research (TMLR)}
}
```

## License

This project is licensed under the MIT License. For the complete terms and conditions, refer to the [LICENSE](LICENSE) file or visit:
[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT).


## Contact

For any questions or concerns, please open an issue or contact Manqing Liu at manqingliu@g.harvard.edu.