"""Tests for benchmark helpers and metadata."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

import quantile_guard


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = spec_from_file_location(module_name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_version_exposed():
    assert isinstance(quantile_guard.__version__, str)
    assert quantile_guard.__version__


def test_benchmark_metadata_uses_package_version():
    bench = _load_module(ROOT / "benchmarks" / "run_linear_baselines.py", "benchmark_run_linear")
    df = pd.DataFrame(
        {
            "model": ["demo"],
            "n": [10],
            "p": [2],
            "n_taus": [7],
            "taus": ["0.1,0.5,0.9"],
            "noise": ["heavy"],
            "fit_time_s": [0.1],
            "predict_time_s": [0.01],
            "mean_pinball_loss": [0.2],
            "crossing_rate": [0.0],
            "empirical_coverage": [0.9],
            "nominal_coverage": [0.8],
            "mean_interval_width": [1.0],
        }
    )
    result = bench.add_metadata(df)
    assert result.loc[0, "package_version"] == quantile_guard.__version__


def test_report_generates_figures_for_actual_quantile_counts(tmp_path):
    report = _load_module(ROOT / "benchmarks" / "report.py", "benchmark_report")
    df = pd.DataFrame(
        [
            {
                "model": "joint",
                "n": 100,
                "p": 5,
                "n_taus": 7,
                "fit_time_s": 0.2,
                "mean_pinball_loss": 0.4,
                "crossing_rate": 0.0,
                "empirical_coverage": 0.9,
            },
            {
                "model": "independent",
                "n": 100,
                "p": 5,
                "n_taus": 7,
                "fit_time_s": 0.1,
                "mean_pinball_loss": 0.41,
                "crossing_rate": 0.2,
                "empirical_coverage": 0.9,
            },
            {
                "model": "joint",
                "n": 200,
                "p": 5,
                "n_taus": 13,
                "fit_time_s": 0.5,
                "mean_pinball_loss": 0.39,
                "crossing_rate": 0.0,
                "empirical_coverage": 0.97,
            },
            {
                "model": "independent",
                "n": 200,
                "p": 5,
                "n_taus": 13,
                "fit_time_s": 0.2,
                "mean_pinball_loss": 0.42,
                "crossing_rate": 0.3,
                "empirical_coverage": 0.97,
            },
        ]
    )

    report.generate_figures(df, tmp_path)

    assert (tmp_path / "fit_time_vs_n.png").exists()
    assert (tmp_path / "pinball_loss_vs_n.png").exists()
    assert (tmp_path / "crossing_rate.png").exists()
    assert (tmp_path / "benchmark_overview.png").exists()


def test_report_metadata_documents_actual_benchmark_shape():
    report = _load_module(ROOT / "benchmarks" / "report.py", "benchmark_report_meta")
    df = pd.DataFrame(
        {
            "model": ["sklearn (independent)", "PDLP sparse (joint, non-crossing)"],
            "n": [500, 1000],
            "p": [10, 10],
            "n_taus": [7, 13],
            "noise": ["heavy", "heavy"],
            "package_version": ["0.6.1", "0.6.1"],
            "python_version": ["3.11.0", "3.11.0"],
            "platform": ["Linux", "Linux"],
        }
    )

    metadata = report.generate_metadata_section(df)

    assert "Quantile counts: 7, 13" in metadata
    assert "n=500/p=10/heavy" in metadata
    assert "n=1,000/p=10/heavy" in metadata
    assert "Package version: 0.6.1" in metadata
