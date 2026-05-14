#!/usr/bin/env python3
"""
Compare qlib cached features.pkl with QuantaAlpha factor recomputation.

Example:
  uv run python scripts/debug_compare_factor_pipelines.py \
    --config configs/backtest.yaml \
    --factor-json /Users/wjh720/Desktop/now/code/qlib/all_factors_library.json \
    --features-pkl /Users/wjh720/Desktop/now/code/QuantaAlpha/features.pkl \
    --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pandas as pd
import yaml


def _install_numpy_pickle_shim() -> None:
    """Compat shim for pickles created under numpy versions using numpy._core.*."""
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    pkg = types.ModuleType("numpy._core")
    pkg.__path__ = []
    sys.modules.setdefault("numpy._core", pkg)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def _install_protobuf_service_shim() -> None:
    """Compat shim for mlflow versions importing google.protobuf.service."""
    if "google.protobuf.service" in sys.modules:
        return

    module = types.ModuleType("google.protobuf.service")

    class Service:  # pragma: no cover - compatibility shim
        pass

    class RpcController:  # pragma: no cover - compatibility shim
        pass

    module.Service = Service
    module.RpcController = RpcController
    sys.modules["google.protobuf.service"] = module


def _read_cached_features(path: Path) -> pd.DataFrame:
    _install_numpy_pickle_shim()
    return pd.read_pickle(path)


def _load_factor_library(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return list(data["factors"].values())


def _build_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def _init_qlib(config: dict) -> None:
    import qlib

    _install_protobuf_service_shim()
    data_cfg = config["data"]
    qlib.init(
        provider_uri=str(Path(data_cfg["provider_uri"]).expanduser()),
        region=data_cfg.get("region", "cn"),
    )


def _recompute_label(config: dict) -> pd.DataFrame:
    from qlib.data import D

    data_cfg = config["data"]
    label_expr = config["dataset"]["label"]
    stocks = D.instruments(data_cfg["market"])
    label_df = D.features(
        stocks,
        [label_expr],
        start_time=data_cfg["start_time"],
        end_time=data_cfg["end_time"],
        freq="day",
    )
    label_df.columns = ["LABEL0"]
    return label_df.sort_index()


def _recompute_factors(config: dict, factors: list[dict]) -> pd.DataFrame:
    from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator

    calculator = CustomFactorCalculator(
        data_df=None,
        cache_dir=Path(config["llm"]["cache_dir"]),
        auto_extract_cache=False,
        config=config,
    )
    return calculator.calculate_factors_batch(factors, use_cache=False, skip_compute=False).sort_index()


def _safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(aligned["left"].corr(aligned["right"], method=method))


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug feature differences between qlib cache and QuantaAlpha")
    parser.add_argument("--config", required=True)
    parser.add_argument("--factor-json", required=True)
    parser.add_argument("--features-pkl", required=True)
    parser.add_argument("--limit", type=int, default=10, help="Compare first N factors sorted by factor_name")
    parser.add_argument("--output", default="", help="Optional CSV path for factor comparison report")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    factor_json_path = Path(args.factor_json).resolve()
    features_pkl_path = Path(args.features_pkl).resolve()

    config = _build_config(config_path)
    _init_qlib(config)

    cached = _read_cached_features(features_pkl_path).sort_index()
    cached_features = cached["feature"]
    cached_label = cached["label"][["LABEL0"]]

    factor_items = sorted(_load_factor_library(factor_json_path), key=lambda item: item["factor_name"])
    factor_items = factor_items[: args.limit]

    print(f"cached features shape: {cached_features.shape}")
    print(f"cached label shape: {cached_label.shape}")
    print(
        "cached date range: "
        f"{cached.index.get_level_values('datetime').min()} -> "
        f"{cached.index.get_level_values('datetime').max()}"
    )

    fresh_label = _recompute_label(config)
    print(
        "recomputed label date range: "
        f"{fresh_label.index.get_level_values('datetime').min()} -> "
        f"{fresh_label.index.get_level_values('datetime').max()}"
    )
    print(f"label corr(cache vs recomputed): {_safe_corr(cached_label['LABEL0'], fresh_label['LABEL0']):.6f}")

    fresh_features = _recompute_factors(config, factor_items)
    print(f"recomputed feature shape: {fresh_features.shape}")

    rows = []
    for factor in factor_items:
        name = factor["factor_name"]
        if name not in cached_features.columns or name not in fresh_features.columns:
            rows.append(
                {
                    "factor_name": name,
                    "status": "missing",
                    "cache_non_null": cached_features[name].notna().sum() if name in cached_features.columns else 0,
                    "fresh_non_null": fresh_features[name].notna().sum() if name in fresh_features.columns else 0,
                }
            )
            continue

        cache_series = cached_features[name]
        fresh_series = fresh_features[name]
        joined = pd.concat([cache_series.rename("cache"), fresh_series.rename("fresh")], axis=1)
        overlap = joined.dropna()
        rows.append(
            {
                "factor_name": name,
                "status": "ok",
                "cache_non_null": int(cache_series.notna().sum()),
                "fresh_non_null": int(fresh_series.notna().sum()),
                "overlap_non_null": int(len(overlap)),
                "pearson_corr": _safe_corr(cache_series, fresh_series),
                "spearman_corr": _safe_corr(cache_series, fresh_series, method="spearman"),
                "cache_mean": float(cache_series.mean()),
                "fresh_mean": float(fresh_series.mean()),
            }
        )

    report = pd.DataFrame(rows).sort_values(
        by=["status", "pearson_corr"],
        ascending=[True, True],
        na_position="first",
    )
    print(report.to_string(index=False))

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
        print(f"saved report: {output_path}")


if __name__ == "__main__":
    main()
