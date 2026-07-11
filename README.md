# Extenics-GNN Differential Framework

Implementation of the GNN-based structural analysis component of the **Extenics-based Graph Difference Framework (EGDF)**.

This repository provides the multi-task GNN training code, inference code, and the annotated dataset used in the paper.

## Repository Structure

```text
EGDF/
├── train.py
├── inference.py
├── data/
│   ├── ecological/
│   │   ├── data/
│   │   └── label/
│   ├── financial/
│   │   ├── data/
│   │   └── label/
│   ├── medical/
│   │   ├── data/
│   │   └── label/
│   └── production/
│       ├── data/
│       └── label/
└── requirements.txt
```

## Code Overview

### `train.py`

Trains the multi-task GNN described in Section 3.2 of the paper. The model jointly predicts:

- node importance
- node problem relevance
- edge conflict probability
- edge conflict severity
- edge problem probability

The script also performs 5-fold cross-validation and trains the final model using a reproducible 60/20/20 train/validation/test split.

### `inference.py`

Loads the trained GNN and analyzes new Extenics basic-element graphs. It outputs ranked key nodes, predicted conflict relations, severity scores, and structured JSON analysis results for subsequent use in the EGDF workflow.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data_dir ./data
```

## Inference

Single-file analysis:

```bash
python inference.py \
  --mode single \
  --domain ecological \
  --data_file ./data/ecological/data/example_data.json \
  --model_path best_model_structural.pt
```

Batch analysis:

```bash
python inference.py \
  --mode batch \
  --domain ecological \
  --data_dir ./data/ecological/data \
  --model_path best_model_structural.pt
```

## Dataset

The dataset contains 200 annotated problem-description graphs across four domains: ecology, finance, medicine, and production. Each domain contains 50 samples represented using Extenics matter-elements, affair-elements (stored as `action_elements`), and relation-elements.

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{cao2026semantic,
  title={From semantic association to structural reasoning: An extenics-GNN differential framework for guiding LLM solution synthesis},
  author={Cao, Tianyi},
  journal={Information Sciences},
  volume={756},
  pages={123868},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.ins.2026.123868}
}
```

## Code and Data

This repository provides the GNN training and inference code together with the full training dataset used in the paper. For the complete EGDF workflow, including the prompt templates and detailed framework configuration, please refer to the appendices of the paper.
