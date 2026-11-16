"""Check the unchanged staging baseline and the exact KCNN improvement commit."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "a8c132ba9975becd2d2bf74ae491cb432527bffa"
SOURCE = "recommenders/models/deeprec/models/pytorch/dkn.py"
TEST = "tests/unit/recommenders/models/test_dkn.py"
SUITE = [TEST] + [
    "tests/unit/recommenders/models/" + name
    for name in ("test_dkn_item2item.py", "test_dkn_dataset.py", "test_fcn_net.py")
]
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)


def restore(revision, path):
    Path(path).write_bytes(subprocess.check_output(["git", "show", revision + ":" + path]))


def check(label, arguments, expected_passed, expected_skipped, expected_failures):
    xml = OUT / (label + ".xml")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments, "-q", "--junitxml=" + str(xml)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    failures = sorted(case.attrib["name"] for case in cases if case.find("failure") is not None)
    errors = [case.attrib["name"] for case in cases if case.find("error") is not None]
    skipped = sum(case.find("skipped") is not None for case in cases)
    passed = len(cases) - len(failures) - len(errors) - skipped
    assert failures == sorted(expected_failures), (label, failures)
    assert not errors, (label, errors)
    assert (passed, skipped) == (expected_passed, expected_skipped), (label, passed, skipped)
    assert result.returncode == (1 if expected_failures else 0)
    return {"name": label, "passed": passed, "skipped": skipped, "failures": failures}


results = []
restore(BASE, SOURCE)
restore(BASE, TEST)
results.append(check("unchanged-staging", SUITE, 79, 1, []))
restore(HEAD, TEST)
results.append(check(
    "new-tests-on-unchanged-staging",
    [TEST, "-k", "projection_order or bounds_projection or full_table_regularization"],
    12, 0,
    ["test_kcnn_bounds_projection_work_by_batch_or_vocabulary[" + case + "]"
     for case in ("127-127", "127-None", "7-127", "127-7")],
))
restore(HEAD, SOURCE)
results.append(check("exact-improvement-head", SUITE, 95, 1, []))
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, TEST], check=True)
subprocess.run(["git", "diff", "--exit-code", "--", SOURCE, TEST], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
(OUT / "summary.json").write_text(json.dumps({"base": BASE, "head": HEAD, "results": results}, indent=2) + "\n")
print("Unchanged baseline, four regressions, exact-head tests and formatting verified.", flush=True)
