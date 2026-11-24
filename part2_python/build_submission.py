"""Build the standalone capstone_part2 submission ZIP from an explicit file set."""
import hashlib
import json
import logging
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def build_submission():
    files = [ROOT / "Metro_Interstate_Traffic_Volume.csv",
             ROOT / "output/pdf/part2_traffic_pipeline_report.pdf"]
    package = ROOT / "part2_python"
    files.extend(package / name for name in (
        "__init__.py", "pipeline.py", "features.py", "visualize.py", "cli.py", "README.md",
        "requirements.txt", "pipeline.log", "features.log", "visualize.log", "cli.log"))
    files.extend(sorted((package / "tests").glob("test_*.py")))
    files.extend(sorted((package / "figures").glob("*.png")))
    files.extend(sorted((package / "figures").glob("*.csv")))
    files.append(package / "figures/figure_metadata.json")
    files.append(package / "validation/cli_acceptance.json")
    files.extend(sorted((ROOT / "data/processed").glob("*.csv")))
    files.extend(sorted((ROOT / "data/processed").glob("*.json")))
    wrappers = {
        "pipeline.py": "part2_python.pipeline",
        "feature_engineering.py": "part2_python.features",
        "visualizations.py": "part2_python.visualize",
        "traffic_app.py": "part2_python.cli",
    }
    folder = ROOT / "output/submission"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "capstone_part2.zip"
    hashes = {}
    readme = """# Capstone Part 2 Traffic Analytics Pipeline

Extract the ZIP and open a terminal in capstone_part2 (the folder containing this README).
Install dependencies: python -m pip install -r part2_python/requirements.txt
Verified environment: Python 3.11.17, NumPy 2.4.6, pandas 3.1.1, Matplotlib 3.11.2.

Run the complete pipeline: python pipeline.py --all
Run time features only: python feature_engineering.py
Run scaled and quartile features: python feature_engineering.py --quartiles
Run figures: python visualizations.py

Three analytics commands:
python traffic_app.py at --time "2017-01-01 00:00:00"
python traffic_app.py summary --start 2017-01-01 --end 2017-01-31
python traffic_app.py compare --start 2017-01-01 --end 2017-01-31

Run checks: python -m unittest discover -s part2_python/tests -v
The root scripts are entry-point aliases for the implementation modules. Keep the
whole folder structure: modules locate raw data and outputs relative to this root.
The full pipeline rewrites generated tables/figures and part2_python/pipeline.log;
it does not modify the raw input. CLI logs are separate and overwritten each run.

Report: part2_traffic_pipeline_report.pdf (two pages).
Figures: part2_python/figures (three PNGs with supporting data).
Sample complete log: part2_python/pipeline.log.
Detailed methods, logging and command semantics: part2_python/README.md.
MANIFEST.json lists SHA-256 values for every other packaged file.

Source repository: https://github.com/FSLiao375426/smart-city-traffic-capstone-003
The repository preserves incremental Git history; the ZIP is a portable snapshot.
Data: Hogue (2019), Metro Interstate Traffic Volume, UCI Machine Learning Repository,
https://doi.org/10.24432/C5X60B (CC BY 4.0).
The labels are traffic-volume proxies, not observed accident risk. Full-period plots
are descriptive; the final modelling split and forecast-time inputs remain undecided.
"""
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        def write(name, data):
            if isinstance(data, str):
                data = data.encode("utf-8")
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr("capstone_part2/" + name, data)
        for source in files:
            name = source.relative_to(ROOT).as_posix()
            if source.suffix == ".pdf":
                name = source.name
            write(name, source.read_bytes())
        for name, module in wrappers.items():
            write(name, f'"""Course-named entry point for {module}."""\nimport logging\nfrom {module} import main\nlogger = logging.getLogger(__name__)\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n')
        write("README.md", readme)
        archive.writestr("capstone_part2/MANIFEST.json", json.dumps(hashes, indent=2) + "\n")
    logger.info("Saved %s with %d manifest entries", output, len(hashes))
    return output


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    handlers = [logging.StreamHandler(), logging.FileHandler(ROOT / "part2_python/build_submission.log", mode="w", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    build_submission()
