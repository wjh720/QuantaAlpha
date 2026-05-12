"""
QuantaAlpha custom workspace.

Overrides rdagent QlibFBWorkspace: project-level factor_template overrides default YAML;
base files (read_exp_res.py, etc.) still from rdagent; init empty git repo in workspace to suppress qlib recorder git output.
"""

import re
import subprocess
from pathlib import Path

from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace as _RdagentQlibFBWorkspace
from rdagent.log import rdagent_logger as logger

from quantaalpha.utils.qlib_data import resolve_qlib_benchmark, resolve_qlib_market, resolve_qlib_provider_uri, resolve_qlib_region

_CUSTOM_TEMPLATE_DIR = Path(__file__).resolve().parent / "factor_template"


class QlibFBWorkspace(_RdagentQlibFBWorkspace):
    """
    Override rdagent QlibFBWorkspace: inject project factor_template/ YAML over defaults;
    init empty git repo in workspace to avoid qlib recorder git help output.
    """

    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(template_folder_path, *args, **kwargs)
        if _CUSTOM_TEMPLATE_DIR.exists():
            self.inject_code_from_folder(_CUSTOM_TEMPLATE_DIR)
            logger.info(f"Overrode rdagent default config with project template: {_CUSTOM_TEMPLATE_DIR}")

    def before_execute(self) -> None:
        """Init empty git repo in workspace to suppress qlib recorder git warnings."""
        super().before_execute()
        self._rewrite_qlib_config_provider()
        git_dir = self.workspace_path / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(self.workspace_path),
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

    def _rewrite_qlib_config_provider(self) -> None:
        provider_uri = resolve_qlib_provider_uri()
        region = resolve_qlib_region(provider_uri=provider_uri)
        market = resolve_qlib_market(region=region, provider_uri=provider_uri)
        benchmark = resolve_qlib_benchmark(region=region, provider_uri=provider_uri)
        for config_name in ("conf_baseline.yaml", "conf_combined_factors.yaml", "conf.yaml"):
            config_path = self.workspace_path / config_name
            if not config_path.exists():
                continue

            text = config_path.read_text(encoding="utf-8")
            text = re.sub(
                r'provider_uri:\s*(".*?"|\S+)',
                f'provider_uri: "{provider_uri}"',
                text,
                count=1,
            )
            text = re.sub(r"region:\s*\S+", f"region: {region}", text, count=1)
            text = re.sub(r"market:\s*&market\s+\S+", f"market: &market {market}", text, count=1)
            text = re.sub(r"benchmark:\s*&benchmark\s+\S+", f"benchmark: &benchmark {benchmark}", text, count=1)
            config_path.write_text(text, encoding="utf-8")
