# ST-DRA: Physics-Grounded Spatio-Temporal Traffic Forecasting

This repository contains the PyTorch implementation of **ST-DRA** (*Decouple, Route, and Align*), a spatio-temporal traffic forecasting framework designed to improve both prediction accuracy and physical interpretability.

ST-DRA contains three coordinated modules:

- **DTD - Dual-Stream Temporal Decoupling** separates traffic observations into slowly varying trend and high-frequency seasonal/transient components, and encodes the two streams independently.
- **DTR - Dynamic Topology Routing** generates a pool of topology experts and performs sparse node-level Top-k routing to model heterogeneous spatial interactions.
- **PCA - Prototype Consistency Alignment** aligns representations from two graph views with learnable traffic prototypes and an anti-collapse regularizer.

The model is trained with the forecasting objective, temporal heterogeneity objective, and prototype alignment objective. The implementation follows the data convention used by the public traffic datasets from the [ST-SSL dataset repository](https://github.com/Echo-Ji/ST-SSL_Dataset), while the model architecture and code in this repository are ST-DRA.

## Repository structure

```text
ST-DRA/
├── configs/                  # Dataset-specific YAML configurations
├── data/                     # Prepared traffic datasets
├── lib/
│   ├── dataloader.py         # Base dataset loader and normalization
│   ├── dataloader_st_dra.py  # ST-DRA data-loader wrapper
│   ├── logger.py
│   ├── metrics.py
│   └── utils.py
├── model/
│   ├── st_dra_model.py       # ST-DRA model and PCA module
│   ├── temporal_decoupling.py # DTD encoder and temporal decomposition
│   ├── layers.py             # Temporal, graph, and DTR layers
│   ├── models.py             # Reference model components
│   └── trainer.py            # Training, validation, and testing
├── main_st_dra.py            # Training/testing entry point
├── requirements.txt
└── README.md
```

## Requirements

The original environment uses Python 3.8 and the following packages:

```text
numpy==1.21.2
pandas==1.3.5
PyYAML==6.0
torch==1.10.1
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The code also runs with compatible newer PyTorch versions. GPU execution is used automatically when PyTorch reports CUDA as available.

## Datasets

The prepared datasets can be obtained from the [ST-SSL_Dataset GitHub repository](https://github.com/Echo-Ji/ST-SSL_Dataset). The supported dataset names are:

```text
NYCBike1
NYCBike2
NYCTaxi
BJTaxi
```

Place the downloaded files under `data/` as follows:

```text
data/
├── NYCBike1/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── adj_mx.npz
├── NYCBike2/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── adj_mx.npz
├── NYCTaxi/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── adj_mx.npz
└── BJTaxi/
    ├── train.npz
    ├── val.npz
    ├── test.npz
    └── adj_mx.npz
```

### Dataset file format

Each `train.npz`, `val.npz`, and `test.npz` contains four NumPy arrays:

| Array | Shape | Description |
|---|---|---|
| `x` | `(#samples, #lookback, #nodes, #flow_types)` | Historical traffic input |
| `y` | `(#samples, #horizon, #nodes, #flow_types)` | Traffic flow to be predicted |
| `x_offsets` | `(#lookback, 1)` | Relative offsets of the input window |
| `y_offsets` | `(#horizon, 1)` | Relative offsets of the prediction window |

The last dimension contains two traffic channels: inflow and outflow. The current datasets use a one-step prediction horizon.

Each `adj_mx.npz` contains:

| Array | Shape | Description |
|---|---|---|
| `adj_mx` | `(#nodes, #nodes)` | Predefined spatial adjacency matrix |

The supplied datasets currently have the following layouts:

| Dataset | Lookback | Nodes | Train / Val / Test samples |
|---|---:|---:|---:|
| NYCBike1 | 19 | 128 | 3023 / 431 / 864 |
| NYCBike2 | 35 | 200 | 1912 / 274 / 546 |
| NYCTaxi | 35 | 200 | 1912 / 274 / 546 |
| BJTaxi | 35 | 1024 | 3814 / 544 / 1090 |

The loader normalizes the traffic arrays using statistics from the training and validation inputs. The model expects tensors in `[batch, time, node, channel]` order.

## Configuration

The default dataset configurations are located in `configs/`.

Important ST-DRA options include:

```yaml
use_dtd: true                  # Dual-Stream Temporal Decoupling
use_dtr: true                  # Dynamic Topology Routing
use_pca: true                  # Prototype Consistency Alignment

```

The DTR expert pool can also be overridden from the command line:

```text
--expert_num   Total number of topology experts
--topk         Number of active experts per node
```

## Training

Run from the project root:

```bash
cd ST-DRA
python main_st_dra.py --config_filename configs/NYCBike1.yaml
```

Other datasets can be selected by changing the configuration file:

```bash
python main_st_dra.py --config_filename configs/NYCBike2.yaml
python main_st_dra.py --config_filename configs/NYCTaxi.yaml
python main_st_dra.py --config_filename configs/BJTaxi.yaml
```

GPU selection is controlled by `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 python main_st_dra.py --config_filename configs/NYCBike1.yaml
```

Useful command-line overrides are:

```bash
python main_st_dra.py \
  --config_filename configs/NYCBike1.yaml \
  --seed 31 \
  --expert_num 4 \
  --topk 2 \
  --log_dir experiments/NYCBike1/stdra_run
```

## Testing a saved checkpoint

When `mode` is set to `test`, provide the checkpoint produced during training:

```bash
python main_st_dra.py \
  --config_filename configs/NYCBike1.yaml \
  --mode test \
  --best_path experiments/NYCBike1/copy_protocol_main/best_model.pth
```

## Experiment outputs

For a normal run with `debug: false`, the configured `log_dir` contains:

```text
best_model.pth       # Best validation checkpoint
stats.pkl            # Epoch-level training and validation losses
source_config.yaml   # Original configuration used for the run
resolved_config.yaml # Configuration after command-line overrides
test_metrics.json    # MAE, MAPE, RMSE, and sample-count statistics
```

For a quick smoke test without GPU allocation, use a small temporary dataset and set `CUDA_VISIBLE_DEVICES` to an empty value. This checks the complete data-loading, training, validation, and testing pipeline but is not intended for reporting model accuracy.

## Citation

If you use this implementation, please cite the ST-DRA work:

```text
Xiaoyan Huang, Zhizhe Lin, Chaozhi Yu, Dan Yang, Teng Zhou, and Chunjie Cao.
Physics-Grounded Spatio-Temporal Traffic Forecasting via Dual-Stream Decoupling,
Sparse Expert Routing, and Prototypical Representation Alignment.
```

The public traffic data format is provided by the [ST-SSL_Dataset repository](https://github.com/Echo-Ji/ST-SSL_Dataset).
