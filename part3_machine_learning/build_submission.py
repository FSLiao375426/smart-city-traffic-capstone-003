"""Build the capstone_part3 review archive from an explicit set of project assets."""
import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PART = ROOT / "part3_machine_learning"
ARCHIVE = ROOT / "output/submission/capstone_part3.zip"
logger = logging.getLogger(__name__)


def selected_files():
    files = [ROOT / "Metro_Interstate_Traffic_Volume.csv"]
    for name in ("__init__.py", "pipeline.py", "features.py", "visualize.py", "requirements.txt"):
        files.append(ROOT / "part2_python" / name)
    for name in ("traffic_supervised.csv", "traffic_model_features.csv", "traffic_hourly.csv",
                 "quartile_values.json", "imputation_values.json", "scaling_values.json"):
        files.append(ROOT / "data/processed" / name)
    for name in ("part1_data_analytics_insights_report.pdf", "part2_traffic_pipeline_report.pdf"):
        files.append(ROOT / "output/pdf" / name)
    for pattern in ("*.py", "*.md", "*.log", "requirements.txt", "tests/*.py", "figures/*.png",
                    "results/**/*.json", "results/**/*.csv", "models/**/*.joblib",
                    "models/**/*.keras", "models/**/*.json"):
        files.extend(PART.glob(pattern))
    return sorted(files)


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with TemporaryDirectory(dir=ARCHIVE.parent) as temp:
        draft = Path(temp) / ARCHIVE.name
        with ZipFile(draft, "w", ZIP_DEFLATED) as archive:
            for file in selected_files():
                name = file.relative_to(ROOT).as_posix()
                data = file.read_bytes()
                hashes[name] = hashlib.sha256(data).hexdigest()
                archive.writestr("capstone_part3/" + name, data)
            template = PART / "submission_guide.md.in"
            readme = (template if template.exists() else ROOT / "README.md").read_bytes()
            archive.writestr("capstone_part3/README.md", readme)
            hashes["README.md"] = hashlib.sha256(readme).hexdigest()
            archive.writestr("capstone_part3/manifest.json", json.dumps(hashes, indent=2) + "\n")
        draft.replace(ARCHIVE)
    logger.info("Packed %d manifest entries in %s", len(hashes), ARCHIVE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
