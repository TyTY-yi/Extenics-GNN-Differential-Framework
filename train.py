import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv


DOMAINS = ["ecological", "financial", "medical", "production"]
DOMAIN_MAPPING = {domain: idx for idx, domain in enumerate(DOMAINS)}


def normalize_relation_type(relation_type: str) -> str:
    """Normalize relation-type aliases consistently for training and inference."""
    relation_type = relation_type.strip()
    aliases = {
        "Conflict": "Conflicts",
    }
    return aliases.get(relation_type, relation_type)


# ============ Preprocessing: relation vocabulary and data validation ============
def extract_relation_types_and_validate(
    base_dir: Path,
    domains: Iterable[str],
) -> Tuple[Dict[str, int], List[dict]]:
    """Scan data files, build the relation vocabulary, and validate label availability."""
    relation_types = set()
    file_info: List[dict] = []
    missing_labels: List[str] = []

    print("\n" + "=" * 70)
    print("Scanning data files...")
    print("=" * 70)

    for domain in domains:
        domain_path = base_dir / domain
        data_dir = domain_path / "data"
        label_dir = domain_path / "label"

        if not data_dir.exists():
            print(f"Warning: {data_dir} does not exist")
            continue

        data_files = sorted(data_dir.glob("*_data.json"))
        print(f"\n{domain}: {len(data_files)} files")

        for data_path in data_files:
            example_name = data_path.name.removesuffix("_data.json")
            label_path = label_dir / f"{example_name}_labels.json"
            has_label = label_path.exists()

            if not has_label:
                missing_labels.append(f"{domain}/{data_path.name}")

            try:
                with data_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                for relation in data.get("relation_elements", []):
                    relation_type = normalize_relation_type(
                        relation.get("relation", "")
                    )
                    if relation_type:
                        relation_types.add(relation_type)

                file_info.append(
                    {
                        "domain": domain,
                        "file": data_path.name,
                        "has_label": has_label,
                        "num_nodes": len(data.get("matter_elements", []))
                        + len(data.get("action_elements", [])),
                        "num_edges": len(data.get("relation_elements", [])),
                    }
                )
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                print(f"  Error reading {data_path.name}: {exc}")

    relation_types = sorted(relation_types)
    relation_mapping = {
        relation_type: idx for idx, relation_type in enumerate(relation_types)
    }
    relation_mapping["<UNK>"] = len(relation_mapping)

    print(f"\nFound {len(relation_types)} known relation types + 1 unknown type:")
    for relation_type, idx in relation_mapping.items():
        print(f"  {idx}: {relation_type}")

    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} files are missing labels:")
        for item in missing_labels[:10]:
            print(f"  - {item}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    available = sum(1 for item in file_info if item["has_label"])
    print(f"\nAvailable for training: {available}/{len(file_info)}")

    return relation_mapping, file_info


# ============ Multi-domain dataset ============
class MultiDomainConflictDataset(Dataset):
    def __init__(
        self,
        base_dir: Path,
        relation_type_mapping: Dict[str, int],
        domains: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.base_dir = Path(base_dir)
        self.domains = list(domains) if domains is not None else list(DOMAINS)
        self.domain_mapping = dict(DOMAIN_MAPPING)
        self.relation_type_mapping = relation_type_mapping
        self.num_relation_types = len(relation_type_mapping)
        self.graphs: List[Data] = []
        self.failed_files: List[dict] = []
        self._load_all_data()

    def _load_all_data(self) -> None:
        print("\n" + "=" * 70)
        print("Loading dataset...")
        print("=" * 70)

        for domain in self.domains:
            if domain not in self.domain_mapping:
                raise ValueError(f"Unknown domain: {domain}")

            domain_path = self.base_dir / domain
            data_dir = domain_path / "data"
            label_dir = domain_path / "label"

            if not data_dir.exists() or not label_dir.exists():
                print(f"Warning: skipping {domain}; expected {data_dir} and {label_dir}")
                continue

            domain_id = self.domain_mapping[domain]
            data_files = sorted(data_dir.glob("*_data.json"))
            loaded_count = 0

            for data_path in data_files:
                example_name = data_path.name.removesuffix("_data.json")
                label_path = label_dir / f"{example_name}_labels.json"

                if not label_path.exists():
                    self.failed_files.append(
                        {
                            "domain": domain,
                            "file": data_path.name,
                            "reason": "Label missing",
                        }
                    )
                    continue

                graph = self._process_single_example(
                    data_path=data_path,
                    label_path=label_path,
                    domain_id=domain_id,
                    domain_name=domain,
                )
                if graph is None:
                    self.failed_files.append(
                        {
                            "domain": domain,
                            "file": data_path.name,
                            "reason": "Processing failed",
                        }
                    )
                    continue

                self.graphs.append(graph)
                loaded_count += 1

            print(f"{domain}: {loaded_count}/{len(data_files)}")

    def _process_single_example(
        self,
        data_path: Path,
        label_path: Path,
        domain_id: int,
        domain_name: str,
    ) -> Optional[Data]:
        try:
            with data_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            with label_path.open("r", encoding="utf-8") as f:
                labels = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"  Failed to parse {data_path.name}: {exc}")
            return None

        node_ids: List[str] = []
        node_types: List[int] = []
        node_id_to_idx: Dict[str, int] = {}
        node_names: List[str] = []

        for matter in data.get("matter_elements", []):
            node_id = matter.get("id")
            if node_id is None or node_id in node_id_to_idx:
                continue
            idx = len(node_ids)
            node_ids.append(node_id)
            node_types.append(0)
            node_id_to_idx[node_id] = idx
            node_names.append(matter.get("name", f"matter_{idx}"))

        for action in data.get("action_elements", []):
            node_id = action.get("id")
            if node_id is None or node_id in node_id_to_idx:
                continue
            idx = len(node_ids)
            node_ids.append(node_id)
            node_types.append(1)
            node_id_to_idx[node_id] = idx
            node_names.append(action.get("action", f"action_{idx}"))

        num_nodes = len(node_ids)
        if num_nodes == 0:
            return None

        # Node features: [one-hot node type (2), one-hot domain (4)]
        x = torch.zeros((num_nodes, 6), dtype=torch.float)
        for idx, node_type in enumerate(node_types):
            x[idx, node_type] = 1.0
            x[idx, 2 + domain_id] = 1.0

        node_importance = torch.zeros(num_nodes, dtype=torch.float)
        node_relevance = torch.zeros(num_nodes, dtype=torch.float)

        node_scores = labels.get("node_scores", {})
        for node_id, idx in node_id_to_idx.items():
            score = node_scores.get(node_id, {})
            node_importance[idx] = float(score.get("importance", 0.0))
            node_relevance[idx] = float(score.get("problem_relevance", 0.0))

        edge_index: List[List[int]] = []
        edge_attr: List[List[float]] = []
        edge_is_problem: List[float] = []
        edge_severity: List[float] = []
        edge_is_conflict: List[float] = []

        edge_scores = labels.get("edge_scores", {})
        unknown_index = self.relation_type_mapping["<UNK>"]

        for relation in data.get("relation_elements", []):
            source = relation.get("source")
            target = relation.get("target")
            if source not in node_id_to_idx or target not in node_id_to_idx:
                continue

            edge_index.append([node_id_to_idx[source], node_id_to_idx[target]])

            relation_type = normalize_relation_type(
                relation.get("relation", "")
            )
            relation_index = self.relation_type_mapping.get(
                relation_type, unknown_index
            )

            edge_vector = [0.0] * (self.num_relation_types + len(DOMAINS))
            edge_vector[relation_index] = 1.0
            edge_vector[self.num_relation_types + domain_id] = 1.0
            edge_attr.append(edge_vector)

            relation_id = relation.get("id")
            score = edge_scores.get(relation_id, {})
            edge_is_problem.append(float(score.get("is_problem", 0.0)))
            edge_severity.append(float(score.get("severity", 0.0)))
            edge_is_conflict.append(float(score.get("is_conflict", 0.0)))

        if not edge_index:
            return None

        return Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            node_importance=node_importance,
            node_relevance=node_relevance,
            edge_is_problem=torch.tensor(edge_is_problem, dtype=torch.float),
            edge_severity=torch.tensor(edge_severity, dtype=torch.float),
            edge_is_conflict=torch.tensor(edge_is_conflict, dtype=torch.float),
            num_nodes=num_nodes,
            domain_id=torch.tensor([domain_id], dtype=torch.long),
            domain_name=domain_name,
            node_ids=node_ids,
            node_names=node_names,
        )

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]


# ============ GNN model ============
class HeterogeneousMultiTaskGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        num_domains: int = 4,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.domain_embedding = nn.Embedding(num_domains, hidden_channels // 4)
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.conv1 = GATConv(
            hidden_channels,
            hidden_channels,
            heads=4,
            edge_dim=edge_dim,
            concat=True,
            dropout=0.2,
        )
        self.conv2 = GATConv(
            hidden_channels * 4,
            hidden_channels,
            heads=4,
            edge_dim=edge_dim,
            concat=True,
            dropout=0.2,
        )
        self.conv3 = GATConv(
            hidden_channels * 4,
            hidden_channels,
            heads=2,
            edge_dim=edge_dim,
            concat=True,
            dropout=0.2,
        )
        self.conv4 = GATConv(
            hidden_channels * 2,
            hidden_channels,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
        )

        self.shared_node_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.node_importance_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )
        self.node_relevance_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

        edge_input_dim = hidden_channels * 2 + edge_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.edge_conflict_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )
        self.edge_severity_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )
        self.edge_problem_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # PyG batches graph-level domain_id as one value per graph.
        domain_id = data.domain_id.view(-1)
        graph_domain_embeddings = self.domain_embedding(domain_id)

        if hasattr(data, "batch") and data.batch is not None:
            node_domain_embeddings = graph_domain_embeddings[data.batch]
        else:
            node_domain_embeddings = graph_domain_embeddings.expand(x.size(0), -1)

        domain_padding = torch.zeros_like(x)
        domain_padding[:, : node_domain_embeddings.size(1)] = node_domain_embeddings
        x = x + domain_padding

        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)

        node_features = self.shared_node_encoder(x)
        node_importance = self.node_importance_head(node_features).squeeze(-1)
        node_relevance = self.node_relevance_head(node_features).squeeze(-1)

        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_features = self.edge_encoder(edge_features)

        return {
            "node_importance": node_importance,
            "node_relevance": node_relevance,
            "edge_is_conflict": self.edge_conflict_head(edge_features).squeeze(-1),
            "edge_severity": self.edge_severity_head(edge_features).squeeze(-1),
            "edge_is_problem": self.edge_problem_head(edge_features).squeeze(-1),
        }


# ============ Training and evaluation ============
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    eps = 1e-6

    if len(loader) == 0:
        raise ValueError("Training loader is empty.")

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(data)

        loss_importance = F.mse_loss(
            outputs["node_importance"], data.node_importance
        )
        loss_relevance = F.mse_loss(
            outputs["node_relevance"], data.node_relevance
        )
        loss_conflict = F.binary_cross_entropy(
            outputs["edge_is_conflict"].clamp(eps, 1.0 - eps),
            data.edge_is_conflict,
        )
        loss_severity = F.mse_loss(
            outputs["edge_severity"], data.edge_severity
        )
        loss_problem = F.binary_cross_entropy(
            outputs["edge_is_problem"].clamp(eps, 1.0 - eps),
            data.edge_is_problem,
        )

        # Equation (15): conflict-identification loss receives weight lambda_c = 2.0.
        loss = (
            loss_importance
            + loss_relevance
            + 2.0 * loss_conflict
            + loss_severity
            + loss_problem
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    num_batches = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)

            importance_mae = F.l1_loss(
                outputs["node_importance"], data.node_importance
            ).item()
            relevance_mae = F.l1_loss(
                outputs["node_relevance"], data.node_relevance
            ).item()
            severity_mae = F.l1_loss(
                outputs["edge_severity"], data.edge_severity
            ).item()

            totals["importance_mae"] += importance_mae
            totals["relevance_mae"] += relevance_mae
            totals["severity_mae"] += severity_mae
            totals["val_loss"] += importance_mae + relevance_mae + severity_mae
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Evaluation loader is empty.")

    return {key: value / num_batches for key, value in totals.items()}


def evaluate_by_domain(
    model: nn.Module,
    dataset: MultiDomainConflictDataset,
    indices: Iterable[int],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    index_list = list(indices)

    for domain in DOMAINS:
        domain_subset = [
            dataset[idx]
            for idx in index_list
            if dataset[idx].domain_name == domain
        ]
        if not domain_subset:
            continue
        loader = DataLoader(domain_subset, batch_size=batch_size, shuffle=False)
        results[domain] = evaluate(model, loader, device)

    return results


# ============ 5-fold cross-validation ============
def run_cross_validation(
    dataset: MultiDomainConflictDataset,
    edge_dim: int,
    device: torch.device,
    n_splits: int = 5,
    epochs: int = 500,
    batch_size: int = 16,
    hidden_dim: int = 128,
    learning_rate: float = 5e-4,
    random_seed: int = 42,
) -> Dict[str, dict]:
    if len(dataset) < n_splits:
        raise ValueError(
            f"Dataset contains {len(dataset)} graphs, fewer than n_splits={n_splits}."
        )

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    indices = np.arange(len(dataset))
    fold_results: List[Dict[str, float]] = []

    print("\n" + "=" * 70)
    print(f"Running {n_splits}-fold cross-validation...")
    print("=" * 70)

    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(indices), start=1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.2,
            random_state=random_seed + fold,
            shuffle=True,
        )

        train_loader = DataLoader(
            [dataset[int(idx)] for idx in train_idx],
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            [dataset[int(idx)] for idx in val_idx],
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            [dataset[int(idx)] for idx in test_idx],
            batch_size=batch_size,
            shuffle=False,
        )

        model = HeterogeneousMultiTaskGNN(
            in_channels=6,
            hidden_channels=hidden_dim,
            edge_dim=edge_dim,
            num_domains=len(DOMAINS),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )

        best_val_loss = float("inf")
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        max_patience = 30

        for epoch in range(1, epochs + 1):
            train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics["val_loss"]
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if best_state_dict is None:
            raise RuntimeError(f"No checkpoint was obtained for fold {fold}.")

        model.load_state_dict(best_state_dict)
        test_metrics = evaluate(model, test_loader, device)
        fold_results.append(test_metrics)

        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Held-out importance MAE: {test_metrics['importance_mae']:.4f}")
        print(f"  Held-out relevance MAE:  {test_metrics['relevance_mae']:.4f}")
        print(f"  Held-out severity MAE:   {test_metrics['severity_mae']:.4f}")
        print(f"  Held-out aggregate loss: {test_metrics['val_loss']:.4f}")

    metric_keys = [
        "importance_mae",
        "relevance_mae",
        "severity_mae",
        "val_loss",
    ]
    cv_summary: Dict[str, dict] = {}

    print("\n" + "=" * 70)
    print(f"{n_splits}-fold cross-validation summary")
    print("=" * 70)

    for key in metric_keys:
        values = [result[key] for result in fold_results]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        cv_summary[key] = {
            "mean": mean_value,
            "std": std_value,
            "values": values,
        }
        print(f"  {key:25s}: {mean_value:.4f} ± {std_value:.4f}")

    output = {
        "n_splits": n_splits,
        "node_feature_dim": 6,
        "summary": {
            key: {"mean": value["mean"], "std": value["std"]}
            for key, value in cv_summary.items()
        },
        "per_fold": fold_results,
    }
    with Path("cv_results_structural.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nCV results saved to: cv_results_structural.json")
    return cv_summary


# ============ Main training workflow ============
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the EGDF multi-task structural GNN."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root directory containing ecological/financial/medical/production folders.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_cv",
        action="store_true",
        help="Skip 5-fold cross-validation and train only the final model.",
    )
    parser.add_argument(
        "--model_output",
        type=Path,
        default=Path("best_model_structural.pt"),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    base_dir = args.data_dir.resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {base_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    relation_mapping, _ = extract_relation_types_and_validate(base_dir, DOMAINS)
    dataset = MultiDomainConflictDataset(base_dir, relation_mapping, DOMAINS)

    if len(dataset) == 0:
        raise RuntimeError(
            "No valid graphs were loaded. Check the directory layout and JSON files."
        )

    print(f"\nTotal loaded: {len(dataset)} graphs")
    domain_counts = defaultdict(int)
    for graph in dataset.graphs:
        domain_counts[graph.domain_name] += 1

    print("\nDomain distribution:")
    for domain in DOMAINS:
        print(f"  {domain}: {domain_counts.get(domain, 0)}")

    edge_dim = len(relation_mapping) + len(DOMAINS)

    if not args.skip_cv:
        run_cross_validation(
            dataset=dataset,
            edge_dim=edge_dim,
            device=device,
            n_splits=5,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            random_seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("Training final model (60% train / 20% validation / 20% test)...")
    print("=" * 70)

    indices = np.arange(len(dataset))
    stratify_labels = np.array(
        [graph.domain_id.item() for graph in dataset.graphs]
    )

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify_labels,
        shuffle=True,
    )
    train_val_stratify = stratify_labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=args.seed,
        stratify=train_val_stratify,
        shuffle=True,
    )

    print(
        f"Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}"
    )

    train_loader = DataLoader(
        [dataset[int(idx)] for idx in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        [dataset[int(idx)] for idx in val_idx],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        [dataset[int(idx)] for idx in test_idx],
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = HeterogeneousMultiTaskGNN(
        in_channels=6,
        hidden_channels=args.hidden_dim,
        edge_dim=edge_dim,
        num_domains=len(DOMAINS),
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Edge feature dim: {edge_dim}")
    print("Node feature dim: 6 (node type + domain encoding)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 30

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        scheduler.step(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Validation loss: {val_loss:.4f}")
            print(
                f"  Importance MAE: {val_metrics['importance_mae']:.4f} | "
                f"Relevance MAE: {val_metrics['relevance_mae']:.4f} | "
                f"Severity MAE: {val_metrics['severity_mae']:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            args.model_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "relation_type_mapping": relation_mapping,
                    "edge_dim": edge_dim,
                    "hidden_dim": args.hidden_dim,
                    "node_feature_dim": 6,
                    "num_domains": len(DOMAINS),
                    "domain_mapping": DOMAIN_MAPPING,
                    "random_seed": args.seed,
                },
                args.model_output,
            )
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\n" + "=" * 70)
    print("Test-set evaluation")
    print("=" * 70)

    checkpoint = torch.load(
        args.model_output,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print("\nOverall test results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    domain_results = evaluate_by_domain(
        model=model,
        dataset=dataset,
        indices=test_idx,
        batch_size=args.batch_size,
        device=device,
    )

    print("\nTest results by domain:")
    for domain, metrics in domain_results.items():
        print(
            f"  {domain}: importance MAE={metrics['importance_mae']:.4f}, "
            f"relevance MAE={metrics['relevance_mae']:.4f}, "
            f"severity MAE={metrics['severity_mae']:.4f}"
        )

    print("\n" + "=" * 70)
    print("Training finished")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.model_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
