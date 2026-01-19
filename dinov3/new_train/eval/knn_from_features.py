#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import one_hot, softmax
from torch.utils.data import DataLoader, TensorDataset

WORK_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(WORK_DIR))

from dinov3.data.adapters import DatasetWithEnumeratedTargets
from dinov3.eval.helpers import write_results
from dinov3.eval.knn import DictKeysModule, _log_and_format_results_dict
from dinov3.eval.metrics import ClassificationMetricType, build_classification_metric
from dinov3.eval.utils import evaluate

logger = logging.getLogger("dinov3.knn_features")

RESULTS_FILENAME = "results-knn.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("kNN evaluation on precomputed features")
    parser.add_argument("--feature-dir", required=True, help="Directory containing feature_cache files")
    parser.add_argument("--output-dir", default=None, help="Directory to write results (default: parent of feature-dir)")
    parser.add_argument("--train-try", type=int, default=0, help="Index for train_features_try{N}.pt")
    parser.add_argument("--train-features", default=None, help="Override train features .pt path")
    parser.add_argument("--train-labels", default=None, help="Override train labels .pt path")
    parser.add_argument("--test-features", default=None, help="Override test features .pt path")
    parser.add_argument("--test-labels", default=None, help="Override test labels .pt path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ks", default="10,20,100,200", help="Comma/space separated k values")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--metric", default="mean_accuracy")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--train-chunk-size", type=int, default=0, help="Chunk size for train features (0 = full)")
    parser.add_argument("--skip-first-nn", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def parse_ks(value: str) -> tuple[int, ...]:
    parts = [p for p in value.replace(",", " ").split() if p]
    if not parts:
        raise ValueError("ks must not be empty")
    return tuple(int(p) for p in parts)


def parse_metric(name: str) -> ClassificationMetricType:
    for metric in ClassificationMetricType:
        if name == metric.name or name == metric.value:
            return metric
    raise ValueError(f"Unknown metric: {name}")


def infer_num_classes(train_labels: torch.Tensor) -> int:
    if train_labels.ndim == 1:
        return int(train_labels.max().item()) + 1
    return int(train_labels.shape[-1])


def make_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    dataset = DatasetWithEnumeratedTargets(dataset, pad_dataset=False, num_replicas=None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


class KnnModuleSingle(nn.Module):
    def __init__(
        self,
        *,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        device: torch.device,
        ks: tuple[int, ...],
        temperature: float,
        num_classes: int,
        skip_first_nn: bool,
        train_chunk_size: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.ks = ks
        self.max_k = max(self.ks) + (1 if skip_first_nn else 0)
        self.temperature = temperature
        self.num_classes = num_classes
        self.skip_first_nn = skip_first_nn
        self.train_chunk_size = train_chunk_size if train_chunk_size > 0 else 0

        if self.train_chunk_size:
            self.train_features = train_features
            self.train_features_T = None
        else:
            self.train_features = None
            self.train_features_T = train_features.to(self.device).T

        self.train_labels = train_labels.to(self.device)

    def _chunked_topk(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.train_features is not None
        num_train = self.train_features.shape[0]
        max_k = min(self.max_k, num_train)

        topk_sims = None
        topk_indices = None
        for start in range(0, num_train, self.train_chunk_size):
            end = min(start + self.train_chunk_size, num_train)
            chunk = self.train_features[start:end].to(self.device)
            sims = torch.mm(features, chunk.T)
            if topk_sims is None:
                current_k = min(max_k, sims.shape[1])
                topk_sims, topk_idx = sims.topk(current_k, dim=1, largest=True, sorted=True)
                topk_indices = topk_idx + start
            else:
                combined_sims = torch.cat([topk_sims, sims], dim=1)
                chunk_indices = torch.arange(start, end, device=self.device).unsqueeze(0).expand(features.shape[0], -1)
                combined_indices = torch.cat([topk_indices, chunk_indices], dim=1)
                current_k = min(max_k, combined_sims.shape[1])
                topk_sims, topk_pos = combined_sims.topk(current_k, dim=1, largest=True, sorted=True)
                topk_indices = combined_indices.gather(1, topk_pos)
        assert topk_sims is not None and topk_indices is not None
        return topk_sims, topk_indices

    def forward(self, features: torch.Tensor):
        features = features.to(self.device)
        if self.train_chunk_size:
            topk_sims, topk_indices = self._chunked_topk(features)
        else:
            assert self.train_features_T is not None
            sims = torch.mm(features, self.train_features_T)
            topk_sims, topk_indices = sims.topk(min(self.max_k, sims.shape[1]), dim=1, largest=True, sorted=True)

        if self.skip_first_nn:
            topk_sims = topk_sims[:, 1:]
            topk_indices = topk_indices[:, 1:]

        neighbors_labels = self.train_labels[topk_indices]
        topk_sims_transform = softmax(topk_sims / self.temperature, dim=1)
        voting_coefficient = topk_sims_transform.unsqueeze(-1)
        if neighbors_labels.ndim == 2:
            neighbors_labels = one_hot(neighbors_labels, num_classes=self.num_classes)
        matmul = neighbors_labels * voting_coefficient
        return {k: torch.sum(matmul[:, :k, :], dim=1) for k in self.ks}


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else feature_dir.parent

    train_features = Path(args.train_features).expanduser().resolve() if args.train_features else None
    train_labels = Path(args.train_labels).expanduser().resolve() if args.train_labels else None
    test_features = Path(args.test_features).expanduser().resolve() if args.test_features else None
    test_labels = Path(args.test_labels).expanduser().resolve() if args.test_labels else None

    if train_features is None:
        train_features = feature_dir / f"train_features_try{args.train_try}.pt"
    if train_labels is None:
        train_labels = feature_dir / f"train_labels_try{args.train_try}.pt"

    if test_features is None:
        test_features = feature_dir / "test_features.pt"
        if not test_features.exists():
            test_features = feature_dir / "val_features.pt"
    if test_labels is None:
        test_labels = feature_dir / "test_labels.pt"
        if not test_labels.exists():
            test_labels = feature_dir / "val_labels.pt"

    return feature_dir, output_dir, train_features, train_labels, test_features, test_labels


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    _, output_dir, train_feat_path, train_label_path, test_feat_path, test_label_path = resolve_paths(args)
    if not train_feat_path.exists():
        raise FileNotFoundError(f"train features not found: {train_feat_path}")
    if not train_label_path.exists():
        raise FileNotFoundError(f"train labels not found: {train_label_path}")
    if not test_feat_path.exists():
        raise FileNotFoundError(f"test features not found: {test_feat_path}")
    if not test_label_path.exists():
        raise FileNotFoundError(f"test labels not found: {test_label_path}")

    logger.info("Loading train features: %s", train_feat_path)
    train_features = torch.load(train_feat_path, map_location="cpu")
    train_labels = torch.load(train_label_path, map_location="cpu")
    logger.info("Loading test features: %s", test_feat_path)
    test_features = torch.load(test_feat_path, map_location="cpu")
    test_labels = torch.load(test_label_path, map_location="cpu")

    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train features/labels length mismatch")
    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError("test features/labels length mismatch")

    normalize = not args.no_normalize
    train_features = train_features.float()
    test_features = test_features.float()
    if normalize:
        train_features = F.normalize(train_features, dim=1)
        test_features = F.normalize(test_features, dim=1)

    num_classes = infer_num_classes(train_labels)
    ks = parse_ks(args.ks)
    max_valid_k = train_features.shape[0] - (1 if args.skip_first_nn else 0)
    ks = sorted(set(min(k, max_valid_k) for k in ks if k > 0))
    if not ks:
        raise ValueError("No valid k values after clipping to train set size")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    metric_type = parse_metric(args.metric)
    metric_collection = build_classification_metric(metric_type, num_classes=num_classes)
    test_loader = make_feature_loader(test_features, test_labels, args.batch_size, args.num_workers)

    knn_model = KnnModuleSingle(
        train_features=train_features,
        train_labels=train_labels,
        device=device,
        ks=tuple(ks),
        temperature=args.temperature,
        num_classes=num_classes,
        skip_first_nn=args.skip_first_nn,
        train_chunk_size=args.train_chunk_size,
    )
    postprocessors = {k: DictKeysModule([k]) for k in ks}
    metrics = {k: metric_collection.clone() for k in ks}

    with torch.inference_mode():
        _, eval_metrics, _ = evaluate(knn_model, test_loader, postprocessors, metrics, device)

    eval_metrics_dict = {k: {m: v.item() * 100.0 for m, v in eval_metrics[k].items()} for k in ks}
    results_dict = _log_and_format_results_dict(eval_metrics_dict, few_shot_n_tries=1)
    write_results(results_dict, str(output_dir), RESULTS_FILENAME)


if __name__ == "__main__":
    main()
