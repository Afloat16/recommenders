"""Validate the validation metric direction against unchanged staging."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "201ca1ac15ae441749a7fad848cdb0a089f97a0d"
TEST = "tests/unit/recommenders/models/test_sli_rec.py"
SOURCE = "recommenders/models/deeprec/models/sequential/pytorch/sequential_base.py"
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)
FAILURES = [
    "test_slirec_checkpoint_direction_and_patience[logloss-scores0-2-3]",
    "test_slirec_checkpoint_direction_and_patience[rmse-scores1-2-3]",
    "test_slirec_checkpoint_direction_and_patience[ndcg@2-scores3-2-3]",
    "test_slirec_checkpoint_direction_and_patience[f1-scores4-1-2]",
    "test_slirec_checkpoint_direction_and_patience[logloss-scores5-1-2]",
    "test_slirec_checkpoint_direction_and_patience[rmse-scores7-2-3]",
    "test_slirec_checkpoint_matches_actual_validation_optimum[logloss]",
    "test_slirec_checkpoint_matches_actual_validation_optimum[rmse]"
]


def content(revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path])


def restore(revision, path):
    Path(path).write_bytes(content(revision, path))


def check(label, select, passed, failures):
    xml = OUT / (label + ".xml")
    args = [sys.executable, "-m", "pytest", TEST, "-q", "--junitxml=" + str(xml)]
    if select:
        args += ["-k", select]
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    failed = sorted(case.attrib["name"] for case in cases if case.find("failure") is not None)
    assert failed == sorted(failures), (label, failed)
    assert not any(case.find("error") is not None or case.find("skipped") is not None for case in cases)
    assert len(cases) - len(failed) == passed, (label, len(cases), failed)
    assert result.returncode == (1 if failures else 0)
    return {"name":label,"passed":passed,"failures":failed}


subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, HEAD], check=True)
for path in [TEST, SOURCE]:
    restore(BASE, path)
results = [check("unchanged-staging", None, 12, [])]
restore(HEAD, TEST)
results.append(check("new-regressions-before", "checkpoint", 3, FAILURES))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", None, 23, []))
for path in [TEST, SOURCE]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", TEST, SOURCE], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [TEST, SOURCE]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", TEST, SOURCE], check=True)
(OUT / "summary.json").write_text(json.dumps({"base":BASE,"head":HEAD,"results":results},indent=2)+"\n")
print("Baseline, eight expected failures, exact improvement and formatting verified.", flush=True)
