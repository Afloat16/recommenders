"""Validate native DKN item-to-item low-precision normalization."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "eb6af96ae54feacc80b31fd3cb71869a55edc1ad"
SOURCE = "recommenders/models/deeprec/models/pytorch/dkn_item2item.py"
TEST = "tests/unit/recommenders/models/test_dkn_item2item.py"
RELATED = ["tests/unit/recommenders/models/test_dkn.py", "tests/unit/recommenders/models/test_dkn_dataset.py"]
FAILURES = ['test_normalization_precision[0.0-float16]', 'test_normalization_precision[0.001-float16]', 'test_normalization_precision_optimizer_step[0.0-False]', 'test_normalization_precision_optimizer_step[0.0-True]', 'test_normalization_precision_optimizer_step[0.001-False]', 'test_normalization_precision_optimizer_step[0.001-True]']
HALF_CASES = sorted([f"test_normalization_precision[{s}-float16]" for s in ["0.0", "0.001", "1.0"]] +
                    [f"test_normalization_precision_optimizer_step[{s}-{a}]" for s in ["0.0", "0.001"] for a in ["False", "True"]])
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)

def content(revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path])

def restore(revision, path):
    Path(path).write_bytes(content(revision, path))

def half_cpu_supported():
    try:
        torch.nn.functional.linear(torch.ones(1, 2, dtype=torch.float16), torch.ones(2, 2, dtype=torch.float16))
        torch.nn.functional.conv2d(torch.ones(1, 1, 2, 2, dtype=torch.float16), torch.ones(1, 1, 1, 1, dtype=torch.float16))
        with torch.autocast("cpu", dtype=torch.float16):
            result = torch.nn.functional.linear(torch.ones(1, 2), torch.ones(2, 2))
        assert result.dtype == torch.float16
        return True
    except RuntimeError as error:
        if "not implemented for" in str(error) or "AutocastCPU only support" in str(error):
            print("Independent CPU half capability probe:", str(error), flush=True)
            return False
        raise

def check(label, paths, passed, failures, skips, select=None):
    xml = OUT / (label + ".xml")
    args = [sys.executable, "-m", "pytest", *paths, "-q", "--junitxml=" + str(xml)]
    if select:
        args += ["-k", select]
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    actual = sorted(x.attrib["name"] for x in cases if x.find("failure") is not None)
    skipped = sorted(x.attrib["name"] for x in cases if x.find("skipped") is not None)
    assert actual == sorted(failures), (label, actual)
    assert skipped == sorted(skips), (label, skipped)
    assert not any(x.find("error") is not None for x in cases)
    assert len(cases) - len(actual) - len(skipped) == passed, (label, len(cases))
    assert result.returncode == (1 if failures else 0)
    return dict(name=label, passed=passed, failures=actual, skipped=skipped)

subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, HEAD], check=True)
for path in [SOURCE, TEST]:
    restore(BASE, path)
results = [check("unchanged-dkn-suites", [TEST, *RELATED], 71, [], ["test_trains_on_the_gpu"])]
half_supported = half_cpu_supported()
restore(HEAD, TEST)
results.append(check("new-regressions-before", [TEST], 10 if half_supported else 9,
                     FAILURES if half_supported else [], [] if half_supported else HALF_CASES,
                     "normalization_precision"))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", [TEST, *RELATED], 87 if half_supported else 80,
                     [], ["test_trains_on_the_gpu"] + ([] if half_supported else HALF_CASES)))
for path in [SOURCE, TEST]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, TEST], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [SOURCE, TEST]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", SOURCE, TEST], check=True)
(OUT / "summary.json").write_text(json.dumps(dict(base=BASE, head=HEAD, results=results),indent=2)+"\n")
print("Native KCNN normalization, dtype reference and optimizer regressions verified.", flush=True)
