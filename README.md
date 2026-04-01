This repository contains the code for the paper "Hypergraph Neural Networks Accelerate MUS Enumeration".

## Requirements

Install Python and the main dependencies used in the scripts:

- PyTorch
- PyTorch Geometric
- PySAT
- Numpy
- Pandas
- Transformers
- tqdm

## Quick Start
### Training

```bash
python src/train.py \
--output_dir runs/train
```

### Evaluation

```bash
python src/evaluate.py \
--model_dir runs/train \
--output_dir runs/evaluate
```

If `--data_path` is not provided, evaluation generates random SAT problems and stores them under the output directory.

### Train + Evaluate Pipeline

The provided [train_eval.sh](train_eval.sh) is a simple example pipeline. It trains a model and then evaluates it on random SAT problems. You can modify the script to use different configurations or datasets.

```bash
bash train_eval.sh
```

## Outputs

Training (`--output_dir`):

- `exp_param.json`: experiment configuration
- `model_param.json`: model hyperparameters
- `model.pth`: latest model checkpoint
- `best_model.pth`: best recent model checkpoint
- `reward_log.csv`: reward statistics per PPO epoch
- `loss_log.csv`: loss statistics per update

Evaluation (`--output_dir`):

- `evaluation_results_on_random_SAT_problems.csv`: per-instance results
- `evaluation_param.json`: evaluation configuration and summary metrics
- `evaluate_log/`: per-instance MUS enumeration logs

## License

This project is released under the PolyForm Noncommercial License 1.0.0. See [LICENSE](LICENSE).

Third-party code under `src/SR` is licensed under the Apache License 2.0. See [src/SR/LICENSE](src/SR/LICENSE).
