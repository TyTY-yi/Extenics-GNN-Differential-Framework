import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch_geometric.data import Data

from train import (
    DOMAIN_MAPPING,
    DOMAINS,
    HeterogeneousMultiTaskGNN,
    normalize_relation_type,
)


class MultiDomainConflictAnalyzer:
    """Run the trained EGDF structural GNN on one or more basic-element graphs."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.device = torch.device(device)
        self.domain_mapping = dict(DOMAIN_MAPPING)
        self.domain_names = {idx: name for name, idx in DOMAIN_MAPPING.items()}

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        required_keys = {
            "model_state_dict",
            "relation_type_mapping",
            "edge_dim",
        }
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise KeyError(
                f"Checkpoint is missing required fields: {sorted(missing_keys)}"
            )

        self.relation_mapping: Dict[str, int] = checkpoint[
            "relation_type_mapping"
        ]
        if "<UNK>" not in self.relation_mapping:
            raise KeyError("Checkpoint relation mapping does not contain '<UNK>'.")

        self.num_relation_types = len(self.relation_mapping)
        edge_dim = int(checkpoint["edge_dim"])
        hidden_dim = int(checkpoint.get("hidden_dim", 128))
        node_feature_dim = int(checkpoint.get("node_feature_dim", 6))
        num_domains = int(checkpoint.get("num_domains", len(DOMAINS)))

        expected_edge_dim = self.num_relation_types + len(DOMAINS)
        if edge_dim != expected_edge_dim:
            raise ValueError(
                "Checkpoint edge dimension is inconsistent with its relation vocabulary: "
                f"edge_dim={edge_dim}, expected={expected_edge_dim}."
            )

        self.model = HeterogeneousMultiTaskGNN(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            edge_dim=edge_dim,
            num_domains=num_domains,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Loaded model from: {model_path}")
        print(f"Relation types: {self.num_relation_types}")
        print(f"Validation loss: {checkpoint.get('val_loss', float('nan')):.4f}")

    def analyze_problem(
        self,
        data_dict: dict,
        domain: str,
        top_k_nodes: int = 10,
        conflict_threshold: float = 0.3,
    ) -> dict:
        if domain not in self.domain_mapping:
            raise ValueError(
                f"Unknown domain '{domain}'. Expected one of {list(self.domain_mapping)}."
            )
        if top_k_nodes <= 0:
            raise ValueError("top_k_nodes must be positive.")
        if not 0.0 <= conflict_threshold <= 1.0:
            raise ValueError("conflict_threshold must be between 0 and 1.")

        graph = self._build_graph(data_dict, domain).to(self.device)

        with torch.no_grad():
            outputs = self.model(graph)

        return self._parse_predictions(
            outputs=outputs,
            graph=graph,
            top_k=top_k_nodes,
            threshold=conflict_threshold,
        )

    def _build_graph(self, data_dict: dict, domain: str) -> Data:
        domain_id = self.domain_mapping[domain]

        node_ids: List[str] = []
        node_types: List[int] = []
        node_id_to_idx: Dict[str, int] = {}
        node_id_to_info: Dict[str, dict] = {}

        for matter in data_dict.get("matter_elements", []):
            node_id = matter.get("id")
            if node_id is None or node_id in node_id_to_idx:
                continue
            idx = len(node_ids)
            node_ids.append(node_id)
            node_types.append(0)
            node_id_to_idx[node_id] = idx
            node_id_to_info[node_id] = {
                "type": "matter",
                "name": matter.get("name", f"matter_{idx}"),
                "features": matter.get("features", {}),
            }

        for action in data_dict.get("action_elements", []):
            node_id = action.get("id")
            if node_id is None or node_id in node_id_to_idx:
                continue
            idx = len(node_ids)
            node_ids.append(node_id)
            node_types.append(1)
            node_id_to_idx[node_id] = idx
            node_id_to_info[node_id] = {
                "type": "action",
                "name": action.get("action", f"action_{idx}"),
                "features": action.get("features", {}),
            }

        num_nodes = len(node_ids)
        if num_nodes == 0:
            raise ValueError("The input graph contains no valid nodes.")

        x = torch.zeros((num_nodes, 6), dtype=torch.float)
        for idx, node_type in enumerate(node_types):
            x[idx, node_type] = 1.0
            x[idx, 2 + domain_id] = 1.0

        edge_index: List[List[int]] = []
        edge_attr: List[List[float]] = []
        edge_info: List[dict] = []
        unknown_index = self.relation_mapping["<UNK>"]

        for relation in data_dict.get("relation_elements", []):
            source = relation.get("source")
            target = relation.get("target")
            if source not in node_id_to_idx or target not in node_id_to_idx:
                continue

            edge_index.append([node_id_to_idx[source], node_id_to_idx[target]])

            relation_type = normalize_relation_type(
                relation.get("relation", "")
            )
            relation_index = self.relation_mapping.get(
                relation_type, unknown_index
            )

            edge_vector = [0.0] * (self.num_relation_types + len(DOMAINS))
            edge_vector[relation_index] = 1.0
            edge_vector[self.num_relation_types + domain_id] = 1.0
            edge_attr.append(edge_vector)

            edge_info.append(
                {
                    "id": relation.get("id", f"edge_{len(edge_info)}"),
                    "type": relation_type or "<UNK>",
                    "source": source,
                    "target": target,
                    "features": relation.get("features", {}),
                }
            )

        if not edge_index:
            raise ValueError("The input graph contains no valid edges.")

        graph = Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            num_nodes=num_nodes,
            domain_id=torch.tensor([domain_id], dtype=torch.long),
        )
        graph.domain_name = domain
        graph.node_ids = node_ids
        graph.node_info = node_id_to_info
        graph.edge_info = edge_info
        graph.batch = torch.zeros(num_nodes, dtype=torch.long)
        return graph

    def _parse_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        graph: Data,
        top_k: int,
        threshold: float,
    ) -> dict:
        importance_scores = outputs["node_importance"].detach().cpu().numpy()
        relevance_scores = outputs["node_relevance"].detach().cpu().numpy()

        # The paper describes Top-K node selection by predicted importance.
        actual_top_k = min(top_k, len(graph.node_ids))
        top_indices = importance_scores.argsort()[-actual_top_k:][::-1]

        key_nodes = []
        for idx in top_indices:
            node_id = graph.node_ids[int(idx)]
            node_info = graph.node_info[node_id]
            key_nodes.append(
                {
                    "id": node_id,
                    "name": node_info["name"],
                    "type": node_info["type"],
                    "importance": float(importance_scores[idx]),
                    "relevance": float(relevance_scores[idx]),
                    "features": node_info["features"],
                }
            )

        conflict_probabilities = (
            outputs["edge_is_conflict"].detach().cpu().numpy()
        )
        severity_scores = outputs["edge_severity"].detach().cpu().numpy()
        problem_probabilities = (
            outputs["edge_is_problem"].detach().cpu().numpy()
        )

        conflicts = []
        all_edges = []
        for idx, edge in enumerate(graph.edge_info):
            source_info = graph.node_info[edge["source"]]
            target_info = graph.node_info[edge["target"]]
            edge_result = {
                "id": edge["id"],
                "source": {
                    "id": edge["source"],
                    "name": source_info["name"],
                },
                "target": {
                    "id": edge["target"],
                    "name": target_info["name"],
                },
                "relation_type": edge["type"],
                "conflict_probability": float(conflict_probabilities[idx]),
                "severity": float(severity_scores[idx]),
                "problem_probability": float(problem_probabilities[idx]),
                "description": edge["features"].get("description", ""),
            }
            all_edges.append(edge_result)
            if edge_result["conflict_probability"] > threshold:
                conflicts.append(edge_result)

        conflicts.sort(key=lambda item: item["severity"], reverse=True)
        all_edges.sort(
            key=lambda item: item["conflict_probability"], reverse=True
        )

        all_nodes = []
        for idx, node_id in enumerate(graph.node_ids):
            node_info = graph.node_info[node_id]
            all_nodes.append(
                {
                    "id": node_id,
                    "name": node_info["name"],
                    "type": node_info["type"],
                    "importance": float(importance_scores[idx]),
                    "relevance": float(relevance_scores[idx]),
                    "features": node_info["features"],
                }
            )
        all_nodes.sort(key=lambda item: item["importance"], reverse=True)

        return {
            "domain": graph.domain_name,
            "num_nodes": int(graph.num_nodes),
            "num_edges": int(graph.num_edges),
            "conflict_threshold": threshold,
            "key_nodes": key_nodes,
            "conflicts": conflicts,
            "all_nodes": all_nodes,
            "all_edges": all_edges,
            "summary": self._generate_summary(key_nodes, conflicts),
        }

    @staticmethod
    def _generate_summary(key_nodes: List[dict], conflicts: List[dict]) -> dict:
        return {
            "total_key_nodes": len(key_nodes),
            "total_detected_conflicts": len(conflicts),
            "average_conflict_severity": (
                sum(item["severity"] for item in conflicts) / len(conflicts)
                if conflicts
                else 0.0
            ),
            "maximum_conflict_severity": (
                max(item["severity"] for item in conflicts)
                if conflicts
                else 0.0
            ),
            "high_severity_conflicts": sum(
                item["severity"] > 0.7 for item in conflicts
            ),
            "average_key_node_importance": (
                sum(item["importance"] for item in key_nodes) / len(key_nodes)
                if key_nodes
                else 0.0
            ),
        }


class BatchAnalyzer:
    def __init__(self, analyzer: MultiDomainConflictAnalyzer) -> None:
        self.analyzer = analyzer

    def analyze_directory(
        self,
        data_dir: Path,
        domain: str,
        output_dir: Path,
        top_k: int = 10,
        conflict_threshold: float = 0.3,
    ) -> dict:
        if not data_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {data_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        data_files = sorted(data_dir.glob("*_data.json"))
        print(f"Batch analyzing {len(data_files)} files from {domain}...")

        results = []
        for data_file in data_files:
            print(f"  Processing {data_file.name}...", end=" ")
            try:
                with data_file.open("r", encoding="utf-8") as f:
                    problem_data = json.load(f)

                analysis = self.analyzer.analyze_problem(
                    problem_data,
                    domain,
                    top_k_nodes=top_k,
                    conflict_threshold=conflict_threshold,
                )

                example_name = data_file.stem.removesuffix("_data")
                analysis_path = output_dir / f"{example_name}_analysis.json"
                with analysis_path.open("w", encoding="utf-8") as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=2)

                results.append(
                    {
                        "file": data_file.name,
                        "status": "success",
                        "conflicts": len(analysis["conflicts"]),
                        "key_nodes": len(analysis["key_nodes"]),
                        "output": str(analysis_path),
                    }
                )
                print("OK")
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                print(f"FAILED: {exc}")
                results.append(
                    {
                        "file": data_file.name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        report = {
            "domain": domain,
            "total_files": len(data_files),
            "successful": sum(item["status"] == "success" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "results": results,
        }
        report_path = output_dir / f"batch_analysis_report_{domain}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(
            f"Completed: {report['successful']}/{report['total_files']} successful. "
            f"Report: {report_path}"
        )
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EGDF multi-domain GNN structural inference."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("best_model_structural.pt"),
    )
    parser.add_argument("--data_file", type=Path)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis_results"),
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--conflict_threshold", type=float, default=0.3)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {args.model_path}")

    analyzer = MultiDomainConflictAnalyzer(
        model_path=args.model_path,
        device=args.device,
    )

    if args.mode == "single":
        if args.data_file is None:
            raise ValueError("--data_file is required in single mode.")
        if not args.data_file.exists():
            raise FileNotFoundError(
                f"Input file does not exist: {args.data_file}"
            )

        with args.data_file.open("r", encoding="utf-8") as f:
            problem_data = json.load(f)

        analysis = analyzer.analyze_problem(
            problem_data,
            args.domain,
            top_k_nodes=args.top_k,
            conflict_threshold=args.conflict_threshold,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "analysis.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        print("\nKey nodes:")
        for node in analysis["key_nodes"]:
            print(
                f"  - {node['name']}: importance={node['importance']:.3f}, "
                f"relevance={node['relevance']:.3f}"
            )

        print(f"\nDetected conflicts: {len(analysis['conflicts'])}")
        for conflict in analysis["conflicts"][:10]:
            print(
                f"  - {conflict['source']['name']} -> "
                f"{conflict['target']['name']}: "
                f"probability={conflict['conflict_probability']:.3f}, "
                f"severity={conflict['severity']:.3f}"
            )

        print(f"\nAnalysis saved to: {output_path}")

    else:
        if args.data_dir is None:
            raise ValueError("--data_dir is required in batch mode.")

        batch_analyzer = BatchAnalyzer(analyzer)
        batch_analyzer.analyze_directory(
            data_dir=args.data_dir,
            domain=args.domain,
            output_dir=args.output_dir / args.domain,
            top_k=args.top_k,
            conflict_threshold=args.conflict_threshold,
        )


if __name__ == "__main__":
    main()
