#!/usr/bin/env python3
"""
Plot cumulative return curves for model-traded assets.
"""

from __future__ import annotations

import copy
import logging
import math
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import pandas as pd

from qlib.contrib.report.analysis_position.parse_position import get_position_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _cumsum_return(series: pd.Series) -> pd.Series:
    return series.fillna(0).cumsum()


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def build_asset_trade_detail(position: dict, report_df: pd.DataFrame, label_data: pd.DataFrame) -> pd.DataFrame:
    """Build full asset-date panel for traded assets."""
    position_df = get_position_data(
        position=copy.deepcopy(position),
        report_normal=report_df,
        label_data=label_data,
    ).reset_index()
    if position_df.empty:
        return pd.DataFrame()

    traded_assets = sorted(position_df["instrument"].dropna().unique())
    asset_panel = label_data.loc[label_data.index.get_level_values("instrument").isin(traded_assets)].reset_index()
    asset_panel = asset_panel.rename(columns={"label": "asset_ret"})

    position_df = position_df.drop(columns=["label"], errors="ignore")
    detail_df = asset_panel.merge(position_df, on=["instrument", "datetime"], how="left")
    detail_df["weight"] = detail_df["weight"].fillna(0)
    detail_df["status"] = detail_df["status"].fillna(0)
    detail_df["model_trade_ret"] = detail_df["asset_ret"].fillna(0) * detail_df["weight"]
    detail_df["is_held"] = detail_df["weight"] > 0
    detail_df = detail_df.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    return detail_df


def build_daily_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily model return and equal-weight traded asset return."""
    if detail_df.empty:
        return pd.DataFrame()

    def _agg_day(day_df: pd.DataFrame) -> pd.Series:
        held_mask = day_df["is_held"]
        held_ret = day_df.loc[held_mask, "asset_ret"]
        traded_asset_ret = held_ret.mean() if len(held_ret) > 0 else 0.0
        return pd.Series(
            {
                "model_ret": day_df["model_trade_ret"].sum(),
                "traded_asset_ret": traded_asset_ret,
                "held_asset_count": int(held_mask.sum()),
                "held_weight_sum": day_df.loc[held_mask, "weight"].sum(),
            }
        )

    daily_df = detail_df.groupby("datetime", sort=True, group_keys=False).apply(_agg_day).reset_index()
    daily_df["cum_model_ret"] = _cumsum_return(daily_df["model_ret"])
    daily_df["cum_traded_asset_ret"] = _cumsum_return(daily_df["traded_asset_ret"])
    return daily_df


def build_asset_info(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize asset-level data needed to inspect and replot traded assets."""
    if detail_df.empty:
        return pd.DataFrame()

    def _agg_asset(asset_df: pd.DataFrame) -> pd.Series:
        held_df = asset_df[asset_df["is_held"]]
        first_date = held_df["datetime"].min() if len(held_df) > 0 else pd.NaT
        last_date = held_df["datetime"].max() if len(held_df) > 0 else pd.NaT
        return pd.Series(
            {
                "start_date": asset_df["datetime"].min(),
                "end_date": asset_df["datetime"].max(),
                "first_held_date": first_date,
                "last_held_date": last_date,
                "held_days": int(asset_df["is_held"].sum()),
                "avg_weight": held_df["weight"].mean() if len(held_df) > 0 else 0.0,
                "max_weight": held_df["weight"].max() if len(held_df) > 0 else 0.0,
                "total_asset_ret": asset_df["asset_ret"].fillna(0).sum(),
                "total_model_trade_ret": asset_df["model_trade_ret"].fillna(0).sum(),
                "plot_file": f"{_sanitize_filename(asset_df['instrument'].iloc[0])}.png",
            }
        )

    asset_info = detail_df.groupby("instrument", sort=True, group_keys=False).apply(_agg_asset).reset_index()
    return asset_info


def plot_overall_curves(daily_df: pd.DataFrame, output_path: Path) -> None:
    """Plot overall cumulative return curves."""
    if daily_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(daily_df["datetime"], daily_df["cum_model_ret"], label="Model cumulative ret", linewidth=2.2)
    ax.plot(
        daily_df["datetime"],
        daily_df["cum_traded_asset_ret"],
        label="Held assets cumulative ret",
        linewidth=1.8,
        alpha=0.9,
    )
    ax.set_title("Model vs Held Assets Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_single_asset(args: tuple[str, pd.DataFrame, str]) -> str:
    asset, asset_df, output_dir = args
    asset_df = asset_df.sort_values("datetime").copy()
    asset_df["cum_asset_ret"] = _cumsum_return(asset_df["asset_ret"])
    asset_df["cum_model_trade_ret"] = _cumsum_return(asset_df["model_trade_ret"])

    fig, (ax_ret, ax_weight) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_ret_right = ax_ret.twinx()

    ret_line_left = ax_ret.plot(
        asset_df["datetime"],
        asset_df["cum_asset_ret"],
        label=f"{asset} cumulative ret",
        linewidth=2.0,
        color="tab:blue",
    )[0]
    ret_line_right = ax_ret_right.plot(
        asset_df["datetime"],
        asset_df["cum_model_trade_ret"],
        label=f"{asset} model traded cumulative ret",
        linewidth=1.8,
        alpha=0.9,
        color="tab:orange",
    )[0]

    ax_ret.set_title(f"{asset} Return vs Model Traded Return")
    ax_ret.set_ylabel(f"{asset} cumulative ret", color="tab:blue")
    ax_ret_right.set_ylabel("Model traded cumulative ret", color="tab:orange")
    ax_ret.tick_params(axis="y", labelcolor="tab:blue")
    ax_ret_right.tick_params(axis="y", labelcolor="tab:orange")
    ax_ret.grid(True, alpha=0.25)
    ax_ret.legend(
        [ret_line_left, ret_line_right],
        [ret_line_left.get_label(), ret_line_right.get_label()],
        loc="upper left",
    )

    ax_weight.plot(
        asset_df["datetime"],
        asset_df["weight"].fillna(0),
        label="Position weight",
        linewidth=1.6,
        color="tab:green",
    )
    ax_weight.fill_between(
        asset_df["datetime"],
        0,
        asset_df["weight"].fillna(0),
        color="tab:green",
        alpha=0.2,
    )
    ax_weight.set_xlabel("Date")
    ax_weight.set_ylabel("Weight")
    ax_weight.grid(True, alpha=0.25)
    ax_weight.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_dir) / f"{_sanitize_filename(asset)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_asset_curves_parallel(detail_df: pd.DataFrame, output_dir: Path, n_jobs: int = 100) -> list[Path]:
    """Plot per-asset curves in parallel."""
    if detail_df.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_frames = [(asset, asset_df.copy(), str(output_dir)) for asset, asset_df in detail_df.groupby("instrument", sort=True)]
    if not grouped_frames:
        return []

    max_workers = max(1, min(int(n_jobs), len(grouped_frames)))
    if max_workers == 1:
        return [Path(_plot_single_asset(item)) for item in grouped_frames]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return [Path(path) for path in executor.map(_plot_single_asset, grouped_frames, chunksize=max(1, math.ceil(len(grouped_frames) / max_workers)))]


def save_asset_trade_outputs(
    position: dict,
    report_df: pd.DataFrame,
    label_data: pd.DataFrame,
    output_dir: Path,
    file_prefix: str,
    n_jobs: int = 100,
) -> dict[str, Path]:
    """Save asset trade detail/daily data and plots."""
    data_outputs = save_asset_trade_data(
        position=position,
        report_df=report_df,
        label_data=label_data,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )
    if not data_outputs:
        return {}

    plot_outputs = plot_saved_asset_trade_outputs(
        output_dir=output_dir,
        file_prefix=file_prefix,
        n_jobs=n_jobs,
    )
    return {**data_outputs, **plot_outputs}


def _asset_trade_paths(output_dir: Path, file_prefix: str) -> dict[str, Path]:
    return {
        "detail_parquet": output_dir / f"{file_prefix}_asset_trade_detail.parquet",
        "daily_parquet": output_dir / f"{file_prefix}_asset_trade_daily.parquet",
        "asset_info_parquet": output_dir / f"{file_prefix}_asset_info.parquet",
        "overall_plot": output_dir / f"{file_prefix}_model_vs_held_assets.png",
        "asset_plot_dir": output_dir / f"{file_prefix}_asset_curves",
    }


def save_asset_trade_data(
    position: dict,
    report_df: pd.DataFrame,
    label_data: pd.DataFrame,
    output_dir: Path,
    file_prefix: str,
) -> dict[str, Path]:
    """Step 1: save all data needed for later plotting."""
    detail_df = build_asset_trade_detail(position=position, report_df=report_df, label_data=label_data)
    if detail_df.empty:
        logger.warning("No traded asset detail available for plotting")
        return {}

    daily_df = build_daily_summary(detail_df)
    asset_info_df = build_asset_info(detail_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _asset_trade_paths(output_dir, file_prefix)

    detail_df.to_parquet(paths["detail_parquet"], index=False)
    daily_df.to_parquet(paths["daily_parquet"], index=False)
    asset_info_df.to_parquet(paths["asset_info_parquet"], index=False)

    return {
        "detail_parquet": paths["detail_parquet"],
        "daily_parquet": paths["daily_parquet"],
        "asset_info_parquet": paths["asset_info_parquet"],
    }


def plot_saved_asset_trade_outputs(output_dir: Path, file_prefix: str, n_jobs: int = 100) -> dict[str, Path]:
    """Step 2: load saved data and generate plots."""
    paths = _asset_trade_paths(output_dir, file_prefix)
    daily_df = pd.read_parquet(paths["daily_parquet"])
    detail_df = pd.read_parquet(paths["detail_parquet"])

    plot_overall_curves(daily_df, paths["overall_plot"])
    plot_asset_curves_parallel(detail_df, paths["asset_plot_dir"], n_jobs=n_jobs)

    return {
        "overall_plot": paths["overall_plot"],
        "asset_plot_dir": paths["asset_plot_dir"],
    }


def main():
    parser = argparse.ArgumentParser(description="Plot saved backtest asset trade curves")
    parser.add_argument(
        "--output-dir",
        default="data/results/backtest_v2_results",
        help="Backtest output directory",
    )
    parser.add_argument(
        "--prefix",
        default="all_factors_library",
        help="File prefix used by backtest output",
    )
    parser.add_argument("--n-jobs", type=int, default=100, help="Parallel asset plotting workers")
    args = parser.parse_args()

    outputs = plot_saved_asset_trade_outputs(
        output_dir=Path(args.output_dir),
        file_prefix=args.prefix,
        n_jobs=args.n_jobs,
    )
    print({key: str(value) for key, value in outputs.items()})


if __name__ == "__main__":
    main()
