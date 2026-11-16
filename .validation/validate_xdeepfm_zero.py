"""Verify xDeepFM's zero-norm gradients and unchanged nonzero behavior."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "a19e80dafb707cd6abb30de08b80da6dbf3040f1"
SOURCE = "recommenders/models/deeprec/models/pytorch/xdeepfm.py"
TEST = "tests/unit/recommenders/models/test_xdeepfm.py"
SUITE = [TEST, "tests/unit/recommenders/models/test_ffm_dataset.py", "tests/unit/recommenders/models/test_fcn_net.py"]
SELECT = "exact_fit or nonzero_rmse or zero_cin or frobenius_norm_keeps or zero_norm_terms"
FAILURES = [
    "test_square_loss_at_exact_fit_has_zero_gradient[" + method + "-" + dtype + "]"
    for method in ("regression", "classification") for dtype in ("float32", "float64")
] + [
    "test_zero_cin_filters_have_finite_regularization_gradient[" + layer + "-" + dtype + "]"
    for layer in ("layer_sizes0", "layer_sizes1") for dtype in ("float32", "float64")
] + [
    "test_zero_norm_terms_keep_real_xdeepfm_optimizer_step_finite[" + term + "]"
    for term in ("exact_fit", "cin_filters")
]
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)


def restore(revision, path):
    Path(path).write_bytes(subprocess.check_output(["git", "show", revision + ":" + path]))


def check(label, arguments, expected_passed, expected_failures):
    xml = OUT / (label + ".xml")
    result = subprocess.run([sys.executable, "-m", "pytest", *arguments, "-q", "--junitxml=" + str(xml)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    assert not any(case.find("error") is not None or case.find("skipped") is not None for case in cases)
    failures = sorted(case.attrib["name"] for case in cases if case.find("failure") is not None)
    assert failures == sorted(expected_failures), (label, failures)
    passed = len(cases) - len(failures)
    assert passed == expected_passed, (label, passed)
    assert result.returncode == (1 if expected_failures else 0)
    return {"name": label, "passed": passed, "failures": failures}


restore(BASE, SOURCE)
restore(BASE, TEST)
results = [check("unchanged-staging", SUITE, 35, [])]
restore(HEAD, TEST)
results.append(check("new-tests-on-staging", [TEST, "-k", SELECT], 4, FAILURES))
restore(HEAD, SOURCE)
results.append(check("exact-improvement-head", SUITE, 49, []))
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, TEST], check=True)
subprocess.run(["git", "diff", "--exit-code", "--", SOURCE, TEST], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
(OUT / "summary.json").write_text(json.dumps({"base": BASE, "head": HEAD, "results": results}, indent=2) + "\n")
print("Baseline, ten expected failures, exact improvement and formatting verified.", flush=True)
