from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import sklearn
import statsmodels

from src.classification import run_classification
from src.config import load_config
from src.eda import run_eda
from src.explanatory import run_explanatory
from src.prepare import load_and_prepare
from src.reporting import build_headline_results, build_output_index
from src.text_coding import run_text_coding
from src.utils import ensure_output_dirs, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete Apple retail support dissertation analysis."
    )
    parser.add_argument(
        "--input",
        default="data/apple_support_survey_data.xlsx",
        help="Path to the anonymised survey workbook.",
    )
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument(
        "--config", default="config/model_config.json", help="Model configuration file."
    )
    parser.add_argument(
        "--coding-map",
        default="appendix_materials/manual_coding_hash_map.csv",
        help="Hashed manual coding map.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = Path(args.output)
    tables, _ = ensure_output_dirs(output_dir)
    config = load_config(args.config)
    data, audit, exclusions = load_and_prepare(args.input, config)
    write_json(tables / "data_quality_audit.json", audit)
    exclusions.to_csv(tables / "exclusion_log.csv", index=False)

    run_eda(data, output_dir)
    run_explanatory(data, output_dir, config)
    run_classification(data, output_dir, config)
    run_text_coding(data, output_dir, args.coding_map)
    build_headline_results(output_dir)
    build_output_index(output_dir)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(tables / "run_environment.json", environment)
    print(f"Analysis complete. Outputs written to: {output_dir.resolve()}")
    print(f"Elapsed seconds: {environment['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    main()
