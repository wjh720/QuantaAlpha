from __future__ import annotations

import os
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CN_DATA_DIR = PROJECT_ROOT / "data" / "cn_data"
DEFAULT_US_DATA_DIR = PROJECT_ROOT / "data" / "us_data"
DEFAULT_CN_DATA_ZIP = PROJECT_ROOT / "data" / "qlib_data_cn_1d_latest.zip"
DEFAULT_US_DATA_ZIP = PROJECT_ROOT / "data" / "qlib_data_us_1d_latest.zip"
DEFAULT_QLIB_REGION = "us"
DEFAULT_QLIB_MARKET = "sp500"
DEFAULT_QLIB_BENCHMARK = "^GSPC"
DEFAULT_CN_MARKET = "csi300"
DEFAULT_CN_BENCHMARK = "SH000300"


def _candidate_paths(raw_path: str | os.PathLike | None) -> list[Path]:
    if raw_path is None:
        return []

    path = Path(raw_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)

    unique_candidates: list[Path] = []
    seen = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate_key)
    return unique_candidates


def _is_qlib_data_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "calendars" / "day.txt").exists()
        and (path / "features").exists()
        and (path / "instruments").exists()
    )


def _extract_qlib_data_zip(zip_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    return target_dir


def infer_qlib_region(provider_uri: str | os.PathLike | None = None) -> str | None:
    for candidate in _candidate_paths(provider_uri):
        name = candidate.name.lower()
        if "cn_data" in name:
            return "cn"
        if "us_data" in name:
            return "us"
    return None


def resolve_qlib_provider_uri(
    provider_uri: str | os.PathLike | None = None,
    *,
    auto_extract: bool = True,
) -> str:
    env_provider_uri = os.environ.get("QLIB_DATA_DIR") or os.environ.get("QLIB_PROVIDER_URI")
    for raw_path in (env_provider_uri, provider_uri):
        for candidate in _candidate_paths(raw_path):
            if _is_qlib_data_dir(candidate):
                return str(candidate.resolve())
            if candidate.is_file() and candidate.suffix == ".zip":
                extract_dir = DEFAULT_US_DATA_DIR if candidate == DEFAULT_US_DATA_ZIP else candidate.with_suffix("")
                if auto_extract:
                    return str(_extract_qlib_data_zip(candidate, extract_dir).resolve())

    for default_dir in (DEFAULT_US_DATA_DIR, DEFAULT_CN_DATA_DIR):
        if _is_qlib_data_dir(default_dir):
            return str(default_dir.resolve())

    default_zip_targets = (
        (DEFAULT_US_DATA_ZIP, DEFAULT_US_DATA_DIR),
        (DEFAULT_CN_DATA_ZIP, DEFAULT_CN_DATA_DIR),
    )
    for default_zip, target_dir in default_zip_targets:
        if default_zip.exists() and auto_extract:
            return str(_extract_qlib_data_zip(default_zip, target_dir).resolve())

    if provider_uri is not None:
        provider_candidates = _candidate_paths(provider_uri)
        if provider_candidates:
            return str(provider_candidates[0])

    return str(DEFAULT_US_DATA_DIR)


def resolve_qlib_region(region: str | None = None, provider_uri: str | os.PathLike | None = None) -> str:
    return (
        os.environ.get("QLIB_REGION")
        or region
        or infer_qlib_region(os.environ.get("QLIB_DATA_DIR") or os.environ.get("QLIB_PROVIDER_URI"))
        or infer_qlib_region(provider_uri)
        or DEFAULT_QLIB_REGION
    ).lower()


def resolve_qlib_market(
    market: str | None = None,
    *,
    region: str | None = None,
    provider_uri: str | os.PathLike | None = None,
) -> str:
    env_market = os.environ.get("QLIB_MARKET")
    if env_market:
        return env_market

    resolved_region = resolve_qlib_region(region, provider_uri)
    if market and market not in {DEFAULT_QLIB_MARKET, DEFAULT_CN_MARKET}:
        return market
    if resolved_region == "cn":
        return DEFAULT_CN_MARKET
    return market or DEFAULT_QLIB_MARKET


def resolve_qlib_benchmark(
    benchmark: str | None = None,
    *,
    region: str | None = None,
    provider_uri: str | os.PathLike | None = None,
) -> str:
    env_benchmark = os.environ.get("QLIB_BENCHMARK")
    if env_benchmark:
        return env_benchmark

    resolved_region = resolve_qlib_region(region, provider_uri)
    if benchmark and benchmark not in {DEFAULT_QLIB_BENCHMARK, DEFAULT_CN_BENCHMARK}:
        return benchmark
    if resolved_region == "cn":
        return DEFAULT_CN_BENCHMARK
    return benchmark or DEFAULT_QLIB_BENCHMARK


def apply_qlib_data_env(config: dict) -> dict:
    data_config = config.setdefault("data", {})
    provider_uri = resolve_qlib_provider_uri(data_config.get("provider_uri"))
    region = resolve_qlib_region(data_config.get("region"), provider_uri)
    market = resolve_qlib_market(data_config.get("market"), region=region, provider_uri=provider_uri)

    data_config["provider_uri"] = provider_uri
    data_config["region"] = region
    data_config["market"] = market

    backtest_section = config.get("backtest", {})
    if isinstance(backtest_section, dict):
        nested_backtest = backtest_section.get("backtest")
        if isinstance(nested_backtest, dict):
            nested_backtest["benchmark"] = resolve_qlib_benchmark(
                nested_backtest.get("benchmark"),
                region=region,
                provider_uri=provider_uri,
            )
    return config


def sanitize_exchange_kwargs(exchange_kwargs: dict | None) -> dict:
    if not exchange_kwargs:
        return {}
    return {key: value for key, value in exchange_kwargs.items() if value is not None}
