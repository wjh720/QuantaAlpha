#!/usr/bin/env python3
"""
Plot cumulative return curves for model-traded assets.
"""

from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import pandas as pd

from qlib.contrib.report.analysis_position.parse_position import get_position_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _cumprod_return(series: pd.Series) -> pd.Series:
    return (1.0 + series.fillna(0)).cumprod() - 1.0


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def build_asset_trade_detail(position: dict, report_df: pd.DataFrame, label_data: pd.DataFrame) -> pd.DataFrame:
    """Build full asset-date panel for traded assets."""
    position_df = get_position_data(
        position=position,
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
    daily_df["cum_model_ret"] = _cumprod_return(daily_df["model_ret"])
    daily_df["cum_traded_asset_ret"] = _cumprod_return(daily_df["traded_asset_ret"])
    return daily_df


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
    asset_df["cum_asset_ret"] = _cumprod_return(asset_df["asset_ret"])
    asset_df["cum_model_trade_ret"] = _cumprod_return(asset_df["model_trade_ret"])

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(asset_df["datetime"], asset_df["cum_asset_ret"], label=f"{asset} cumulative ret", linewidth=2.0)
    ax.plot(
        asset_df["datetime"],
        asset_df["cum_model_trade_ret"],
        label=f"{asset} model traded cumulative ret",
        linewidth=1.8,
        alpha=0.9,
    )
    ax.set_title(f"{asset} Return vs Model Traded Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_dir) / f"{_sanitize_filename(asset)}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_asset_curves_parallel(detail_df: pd.DataFrame, output_dir: Path, n_jobs: int = 4) -> list[Path]:
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
    n_jobs: int = 4,
) -> dict[str, Path]:
    """Save asset trade detail/daily data and plots."""
    detail_df = build_asset_trade_detail(position=position, report_df=report_df, label_data=label_data)
    if detail_df.empty:
        logger.warning("No traded asset detail available for plotting")
        return {}

    daily_df = build_daily_summary(detail_df)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / f"{file_prefix}_asset_trade_detail.csv"
    daily_path = output_dir / f"{file_prefix}_asset_trade_daily.csv"
    overall_plot_path = output_dir / f"{file_prefix}_model_vs_held_assets.png"
    asset_plot_dir = output_dir / f"{file_prefix}_asset_curves"

    detail_df.to_csv(detail_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    plot_overall_curves(daily_df, overall_plot_path)
    plot_asset_curves_parallel(detail_df, asset_plot_dir, n_jobs=n_jobs)

    return {
        "detail_csv": detail_path,
        "daily_csv": daily_path,
        "overall_plot": overall_plot_path,
        "asset_plot_dir": asset_plot_dir,
    }
