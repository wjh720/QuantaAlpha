#!/usr/bin/env python3
"""
Plot per-asset curves for saved long holdings parquet.

For each asset, draw:
1. Asset cumulative return over the asset's holding window.
2. Cumulative model_ret on the dates the asset appears in long holdings.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _init_qlib(config: dict) -> None:
    import qlib

    provider_uri = (
        os.environ.get("QLIB_DATA_DIR")
        or os.environ.get("QLIB_PROVIDER_URI")
        or config["data"]["provider_uri"]
    )
    provider_uri = os.path.expanduser(provider_uri)
    region = config["data"].get("region", "cn")
    qlib.init(provider_uri=provider_uri, region=region)


def _normalize_datetime_instrument_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        return df

    names = list(df.index.names)
    if names == ["datetime", "instrument"]:
        return df.sort_index()
    if names == ["instrument", "datetime"]:
        return df.swaplevel().sort_index()

    level0 = df.index.get_level_values(0)
    level1 = df.index.get_level_values(1)
    if pd.api.types.is_datetime64_any_dtype(level0):
        new_index = pd.MultiIndex.from_arrays([level0, level1], names=["datetime", "instrument"])
    else:
        new_index = pd.MultiIndex.from_arrays([level1, level0], names=["datetime", "instrument"])
    out = df.copy()
    out.index = new_index
    return out.sort_index()


def _fetch_asset_returns(config: dict, assets: list[str], start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    from qlib.data import D

    ret_df = D.features(
        assets,
        ["Ref($close, -1)/$close - 1"],
        start_time=start_time.strftime("%Y-%m-%d"),
        end_time=end_time.strftime("%Y-%m-%d"),
        freq="day",
    )
    ret_df.columns = ["asset_ret"]
    return _normalize_datetime_instrument_index(ret_df).reset_index()


def _plot_one_asset(task: tuple[str, pd.DataFrame, pd.DataFrame, str]) -> str:
    asset, pos_df, ret_df, output_dir = task

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_df = pos_df.sort_values("date").copy()
    ret_df = ret_df.sort_values("date").copy()
    if pos_df.empty or ret_df.empty:
        return asset

    first_date = pos_df["date"].min()
    last_date = pos_df["date"].max()
    ret_df = ret_df[(ret_df["date"] >= first_date) & (ret_df["date"] <= last_date)].copy()
    if ret_df.empty:
        return asset

    ret_df["asset_cumret"] = (1 + ret_df["asset_ret"].fillna(0)).cumprod() - 1
    pos_df["model_ret_cum"] = pos_df["model_ret"].fillna(0).cumsum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ret_df["date"], ret_df["asset_cumret"], label="Asset CumRet", linewidth=1.8)
    ax.plot(pos_df["date"], pos_df["model_ret_cum"], label="Holding-Day Model Ret CumSum", linewidth=1.5)
    ax.set_title(asset)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_dir) / f"{asset}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return asset


def run_plot_long_asset_curves(
    positions_parquet: str | Path,
    config_path: str | Path = "configs/backtest.yaml",
    output_dir: str | Path | None = None,
    n_procs: int = 100,
) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = _load_config(config_path)
    _init_qlib(config)

    positions_path = Path(positions_parquet)
    if not positions_path.is_absolute():
        positions_path = Path.cwd() / positions_path
    if not positions_path.exists():
        raise FileNotFoundError(f"Positions parquet not found: {positions_path}")

    output_dir = Path(output_dir) if output_dir else positions_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    positions_df = pd.read_parquet(positions_path)
    positions_df["date"] = pd.to_datetime(positions_df["date"])
    if "model_ret" not in positions_df.columns:
        raise ValueError("positions parquet must contain model_ret column")

    positions_df = positions_df.sort_values(["instrument", "date"]).copy()
    assets = positions_df["instrument"].dropna().unique().tolist()
    if not assets:
        raise ValueError("No instruments found in positions parquet")

    ret_df = _fetch_asset_returns(
        config,
        assets,
        start_time=positions_df["date"].min(),
        end_time=positions_df["date"].max(),
    )
    ret_df["date"] = pd.to_datetime(ret_df["datetime"])
    ret_df = ret_df[["date", "instrument", "asset_ret"]]

    tasks = []
    for asset in assets:
        asset_pos = positions_df[positions_df["instrument"] == asset][["date", "model_ret", "weight"]].copy()
        asset_ret = ret_df[ret_df["instrument"] == asset][["date", "asset_ret"]].copy()
        tasks.append((asset, asset_pos, asset_ret, str(output_dir)))

    worker_count = max(1, min(n_procs, len(tasks)))
    if worker_count == 1:
        for task in tasks:
            _plot_one_asset(task)
    else:
        with mp.Pool(processes=worker_count) as pool:
            list(pool.imap_unordered(_plot_one_asset, tasks))

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-asset long holding curves from parquet.")
    parser.add_argument(
        "--positions-parquet",
        required=True,
        help="Path to <prefix>_long_positions.parquet",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/backtest.yaml",
        help="Backtest config path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-asset plots; defaults to sibling folder next to parquet",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes",
    )
    args = parser.parse_args()

    output_dir = run_plot_long_asset_curves(
        positions_parquet=args.positions_parquet,
        config_path=args.config,
        output_dir=args.output_dir,
        n_procs=args.n_procs,
    )
    print(f"Saved asset plots to {output_dir}")


if __name__ == "__main__":
    main()
