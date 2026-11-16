"""Validate the independent SLi-Rec numerical fixes against unchanged staging."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import torch

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
TEST = "tests/unit/recommenders/models/test_sli_rec.py"
PREFIX = "recommenders/models/deeprec/models/sequential/pytorch/"
CASES = [
    {
        "name": "half-mask", "head": "8686ca9726be9f617f0ea790bbf3d3ea8fd2c0c2",
        "source": PREFIX + "sli_rec.py", "select": "representable or autocast",
        "controls": 10, "after": 26,
        "failures": [
            "test_attention_mask_is_representable_in_score_dtype[0-float16]",
            "test_attention_mask_is_representable_in_score_dtype[3-float16]",
            "test_attention_mask_is_representable_in_score_dtype[5-float16]",
            "test_slirec_cpu_autocast_forward_and_training_step[float16]",
        ],
    },

]
OUT = Path(".validation/results/half-mask")
OUT.mkdir(parents=True, exist_ok=True)


def supports(operation):
    try:
        return operation()
    except RuntimeError:
        return False


def half_linear():
    probe = torch.ones(2, 2, dtype=torch.float16, requires_grad=True)
    torch.nn.functional.linear(probe, probe).sum().backward()
    return True


def half_autocast():
    with torch.autocast("cpu", dtype=torch.float16):
        result = torch.nn.functional.linear(torch.ones(2, 2), torch.ones(2, 2))
    return result.dtype == torch.float16


def half_log_softmax():
    probe = torch.ones(1, 2, dtype=torch.float16, requires_grad=True)
    torch.log_softmax(probe, dim=-1).sum().backward()
    return True


capabilities = {"half_linear": supports(half_linear), "half_autocast": supports(half_autocast), "half_log_softmax": supports(half_log_softmax)}
print("Independent CPU capabilities: " + json.dumps(capabilities), flush=True)
CASES[0]["skipped"] = []
if not capabilities["half_linear"]:
    CASES[0]["skipped"] += CASES[0]["failures"][:3]
if not capabilities["half_autocast"]:
    CASES[0]["skipped"] += CASES[0]["failures"][3:]
for case in CASES:
    case["failures"] = [name for name in case["failures"] if name not in case["skipped"]]
    case["after"] -= len(case["skipped"])


def content(revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path])


def restore(revision, path):
    Path(path).write_bytes(content(revision, path))


def check(label, select, expected_passed, expected_failures, expected_skipped=()):
    xml = OUT / (label + ".xml")
    args = [sys.executable, "-m", "pytest", TEST, "-q", "--junitxml=" + str(xml)]
    if select:
        args += ["-k", select]
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    failures = sorted(case.attrib["name"] for case in cases if case.find("failure") is not None)
    assert not any(case.find("error") is not None for case in cases)
    skipped = sorted(case.attrib["name"] for case in cases if case.find("skipped") is not None)
    assert skipped == sorted(expected_skipped), (label, skipped)
    assert failures == sorted(expected_failures), (label, failures)
    passed = len(cases) - len(failures) - len(skipped)
    assert passed == expected_passed, (label, passed)
    assert result.returncode == (1 if expected_failures else 0)
    return {"name": label, "passed": passed, "failures": failures, "skipped": skipped}


subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, *[case["head"] for case in CASES]], check=True)
paths = [TEST] + [case["source"] for case in CASES]
for path in paths:
    restore(BASE, path)
results = [check("unchanged-staging", None, 12, [])]
for case in CASES:
    for path in paths:
        restore(BASE, path)
    restore(case["head"], TEST)
    results.append(check(case["name"] + "-before", case["select"], case["controls"], case["failures"], case["skipped"]))
    restore(case["head"], case["source"])
    for path in (TEST, case["source"]):
        assert Path(path).read_bytes() == content(case["head"], path)
    results.append(check(case["name"] + "-after", None, case["after"], [], case["skipped"]))
    subprocess.run([sys.executable, "-m", "black", "--check", TEST, case["source"]], check=True)
    subprocess.run(["git", "diff", "--check"], check=True)
for path in paths:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", "--", *paths], check=True)
(OUT / "summary.json").write_text(json.dumps({"base": BASE, "torch": torch.__version__, "capabilities": capabilities, "cases": CASES, "results": results}, indent=2) + "\n")
print("Baseline, the exact contribution head and all expected regressions verified.", flush=True)
