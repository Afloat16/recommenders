"""Verify field validation before flattened FFM bag construction."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "bb40866a7747ffbf2c95021b22ab6b4f488d20d1"
SOURCE = "recommenders/models/deeprec/io/ffm_dataset.py"
TESTS = ["tests/unit/recommenders/models/test_ffm_dataset.py", "tests/unit/recommenders/models/test_xdeepfm.py"]
FAILURES = [f"test_rejects_fields_that_alias_another_instance[{row}-{field}]"
            for row, field in [(0, 5), (0, 9), (1, 0), (1, 5), (2, 0), (2, -3)]] + [
                "test_training_rejects_cross_instance_fields[0-5]",
                "test_training_rejects_cross_instance_fields[1-0]",
            ]
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
for path in [SOURCE, *TESTS]:
    restore(BASE, path)
results = [check("unchanged-ffm-model-suites", 27, [])]
for path in TESTS:
    restore(HEAD, path)
results.append(check("new-field-cases-before", 5, FAILURES,
                     "alias_another_instance or cross_instance_fields or valid_sparse_fields"))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", 40, []))
for path in [SOURCE, *TESTS]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, *TESTS], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [SOURCE, *TESTS]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", SOURCE, *TESTS], check=True)
(OUT / "summary.json").write_text(json.dumps(dict(base=BASE, head=HEAD, results=results), indent=2) + "\n")
print("Per-instance rejection, native training and valid sparse predictions/gradients verified.", flush=True)
