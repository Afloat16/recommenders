"""Validate independent patches against exactly the same staging baseline."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
SOURCE = "recommenders/models/lightgbm/lightgbm_utils.py"
RESULTS = Path(".validation/results")
RESULTS.mkdir(parents=True, exist_ok=True)
CORE = [
    "tests/unit/recommenders/datasets/test_pandas_df_utils.py",
    "tests/unit/recommenders/datasets/test_python_splitter.py",
    "tests/unit/recommenders/datasets/test_split_columns.py",
    "tests/unit/recommenders/evaluation/test_python_evaluation.py",
    "tests/unit/recommenders/utils/test_python_utils.py",
]
REFIT = "tests/unit/recommenders/models/test_numencoder_refit.py"
UNSEEN = "tests/unit/recommenders/models/test_numencoder_unseen_categories.py"
KNOWN_FAILURE = "test_python_ranking_metrics_by_threshold"


def run_pytest(label, paths, expected_failed, expected_total):
    xml = RESULTS / (label + ".xml")
    command = [sys.executable, "-m", "pytest", "-q", *paths, "--junitxml=" + str(xml)]
    result = subprocess.run(command, text=True, capture_output=True)
    output = result.stdout + result.stderr
    (RESULTS / (label + ".txt")).write_text(output)
    print(output, flush=True)
    assert xml.exists(), "pytest did not produce its report"
    root = ET.parse(xml).getroot()
    cases = root.findall(".//testcase")
    failed = {case.attrib["name"] for case in cases if case.find("failure") is not None}
    assert not root.findall(".//error"), "collection or setup failed"
    assert not root.findall(".//skipped"), "unexpected skipped coverage"
    assert len(cases) == expected_total, (label, len(cases), expected_total)
    assert failed == set(expected_failed), (label, failed, expected_failed)
    assert result.returncode == (1 if failed else 0), result.returncode
    return {"tests": len(cases), "failed": sorted(failed),
            "passed": len(cases) - len(failed)}


versions = {name: importlib.metadata.version(name) for name in (
    "numpy", "pandas", "scipy", "scikit-learn", "category-encoders", "pytest", "black"
)}
print(json.dumps({"python": sys.version, "packages": versions}), flush=True)
summary = {"runtime": versions, "python": sys.version, "base": BASE, "results": {}}
subprocess.run(["git", "checkout", BASE, "--", SOURCE], check=True)
summary["results"]["unchanged_core"] = run_pytest(
    "unchanged-core", CORE, {KNOWN_FAILURE}, 89
)
baseline_failures = {
    "test_repeated_numencoder_fit_preserves_features[columns0]",
    "test_repeated_numencoder_fit_preserves_features[columns1]",
    "test_numencoder_refit_matches_fresh_training_and_inference[columns0]",
    "test_numencoder_refit_matches_fresh_training_and_inference[columns1]",
    "test_unseen_categories_keep_distinct_zero_binary_code[1]",
    "test_unseen_categories_keep_distinct_zero_binary_code[3]",
    "test_unseen_categories_keep_distinct_zero_binary_code[7]",
    "test_entirely_unseen_batch_preserves_fitted_width",
    "test_unseen_category_does_not_change_other_field_encoding",
}
summary["results"]["unchanged_regressions"] = run_pytest(
    "unchanged-regressions", [REFIT, UNSEEN], baseline_failures, 15
)
for label, head, path, count in (
    ("refit", "0c31efef1db29ed3b2bfdc8f7d9a9be022e261a6", REFIT, 6),
    ("unseen", "199a65d74bfd856d381adc9bd977e454d5c1df9d", UNSEEN, 9),
):
    subprocess.run(["git", "checkout", head, "--", SOURCE, path, "tests/test_groups.yml"],
                   check=True)
    summary["results"][label] = {
        "head": head,
        "focused": run_pytest(label + "-focused", [path], set(), count),
        "core": run_pytest(label + "-core", CORE, {KNOWN_FAILURE}, 89),
    }
    subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, path], check=True)
    subprocess.run(["git", "diff", "--check"], check=True)
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
