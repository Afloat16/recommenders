"""Validate the independent SLi-Rec numerical fixes against unchanged staging."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
TEST = "tests/unit/recommenders/models/test_sli_rec.py"
PREFIX = "recommenders/models/deeprec/models/sequential/pytorch/"
CASES = [
    {
        "name": "half-mask", "head": "adb9731c857e13673fb15b8cac00d026a57b0022",
        "source": PREFIX + "sli_rec.py", "select": "representable or autocast",
        "controls": 10, "after": 26,
        "failures": [
            "test_attention_mask_is_representable_in_score_dtype[0-float16]",
            "test_attention_mask_is_representable_in_score_dtype[3-float16]",
            "test_attention_mask_is_representable_in_score_dtype[5-float16]",
            "test_slirec_cpu_autocast_forward_and_training_step[float16]",
        ],
    },
    {
        "name": "log-softmax", "head": "4bfd995c9881ade32ab8d420e3f2dba1423bc4a9",
        "source": PREFIX + "sequential_base.py", "select": "pairwise_loss",
        "controls": 8, "after": 25,
        "failures": [
            "test_pairwise_loss_and_gradients_remain_finite[20.0-float16]",
            "test_pairwise_loss_and_gradients_remain_finite[1000.0-float16]",
            "test_pairwise_loss_and_gradients_remain_finite[1000.0-float32]",
            "test_pairwise_loss_and_gradients_remain_finite[1000.0-float64]",
            "test_pairwise_loss_supports_real_slirec_optimization",
        ],
    },
]
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)


def content(revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path])


def restore(revision, path):
    Path(path).write_bytes(content(revision, path))


def check(label, select, expected_passed, expected_failures):
    xml = OUT / (label + ".xml")
    args = [sys.executable, "-m", "pytest", TEST, "-q", "--junitxml=" + str(xml)]
    if select:
        args += ["-k", select]
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    failures = sorted(case.attrib["name"] for case in cases if case.find("failure") is not None)
    assert not any(case.find("error") is not None or case.find("skipped") is not None for case in cases)
    assert failures == sorted(expected_failures), (label, failures)
    passed = len(cases) - len(failures)
    assert passed == expected_passed, (label, passed)
    assert result.returncode == (1 if expected_failures else 0)
    return {"name": label, "passed": passed, "failures": failures}


subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, *[case["head"] for case in CASES]], check=True)
paths = [TEST] + [case["source"] for case in CASES]
for path in paths:
    restore(BASE, path)
results = [check("unchanged-staging", None, 12, [])]
for case in CASES:
    for path in paths:
        restore(BASE, path)
    restore(case["head"], TEST)
    results.append(check(case["name"] + "-before", case["select"], case["controls"], case["failures"]))
    restore(case["head"], case["source"])
    for path in (TEST, case["source"]):
        assert Path(path).read_bytes() == content(case["head"], path)
    results.append(check(case["name"] + "-after", None, case["after"], []))
    subprocess.run([sys.executable, "-m", "black", "--check", TEST, case["source"]], check=True)
    subprocess.run(["git", "diff", "--check"], check=True)
for path in paths:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", "--", *paths], check=True)
(OUT / "summary.json").write_text(json.dumps({"base": BASE, "cases": CASES, "results": results}, indent=2) + "\n")
print("Baseline, both exact contribution heads and all expected regressions verified.", flush=True)
