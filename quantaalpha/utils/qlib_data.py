from __future__ import annotations

import os
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_US_DATA_DIR = PROJECT_ROOT / "data" / "us_data"
DEFAULT_US_DATA_ZIP = PROJECT_ROOT / "data" / "qlib_data_us_1d_latest.zip"
DEFAULT_QLIB_REGION = "us"
DEFAULT_QLIB_MARKET = "sp500"
DEFAULT_QLIB_BENCHMARK = "^GSPC"


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

    if _is_qlib_data_dir(DEFAULT_US_DATA_DIR):
        return str(DEFAULT_US_DATA_DIR.resolve())

    if DEFAULT_US_DATA_ZIP.exists() and auto_extract:
        return str(_extract_qlib_data_zip(DEFAULT_US_DATA_ZIP, DEFAULT_US_DATA_DIR).resolve())

    if provider_uri is not None:
        provider_candidates = _candidate_paths(provider_uri)
        if provider_candidates:
            return str(provider_candidates[0])

    return str(DEFAULT_US_DATA_DIR)


def resolve_qlib_region(region: str | None = None) -> str:
    return (os.environ.get("QLIB_REGION") or region or DEFAULT_QLIB_REGION).lower()


def sanitize_exchange_kwargs(exchange_kwargs: dict | None) -> dict:
    if not exchange_kwargs:
        return {}
    return {key: value for key, value in exchange_kwargs.items() if value is not None}
