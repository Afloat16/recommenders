"""Verify native GeoIMC optimization with sparse features."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "72b22acedfe747e65e8978f5a67583ef7413fc40"
SOURCES = ["recommenders/models/geoimc/geoimc_algorithm.py", "recommenders/models/geoimc/geoimc_data.py"]
TESTS = ["tests/unit/recommenders/models/test_geoimc.py"]
FAILURES = [f"{name}[{row}-{col}]" for name in [
    "test_geoimc_sparse_feature_objective_and_gradients", "test_geoimc_sparse_features_fit_and_cold_start"]
    for row, col in [(True, False), (False, True), (True, True)]]
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)


def content(revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path])


def restore(revision, path):
    Path(path).write_bytes(content(revision, path))


def check(label, passed, failures, select=None):
    xml = OUT / (label + ".xml")
    args = [sys.executable, "-m", "pytest", *TESTS, "-q", "--junitxml=" + str(xml)]
    if select:
        args += ["-k", select]
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    actual = sorted(x.attrib["name"] for x in cases if x.find("failure") is not None)
    assert actual == sorted(failures), (label, actual)
    assert not any(x.find("error") is not None or x.find("skipped") is not None for x in cases)
    assert len(cases) - len(actual) == passed, (label, len(cases))
    assert result.returncode == (1 if failures else 0)
    return dict(label=label, passed=passed, failures=actual, skipped=0)


subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, HEAD], check=True)
for path in [*SOURCES, *TESTS]:
    restore(BASE, path)
results = [check("unchanged-geoimc-suite", 12, [])]
for path in TESTS:
    restore(HEAD, path)
results.append(check("sparse-feature-cases-before", 2, FAILURES,
                     "sparse_feature"))
for path in SOURCES:
    restore(HEAD, path)
results.append(check("exact-head-after", 20, []))
for path in [*SOURCES, *TESTS]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", *SOURCES, *TESTS], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [*SOURCES, *TESTS]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", *SOURCES, *TESTS], check=True)
(OUT / "summary.json").write_text(json.dumps(dict(base=BASE, head=HEAD, results=results), indent=2) + "\n")
print("Sparse feature objective, finite differences, native optimization and cold-start inference verified.", flush=True)
