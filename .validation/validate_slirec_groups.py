"""Validate ranking groups across evaluation batches."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "8325924934e574c081031cf8be91bfe24f5a720c"
TEST = "tests/unit/recommenders/models/test_sli_rec.py"
SOURCE = "recommenders/models/deeprec/models/sequential/pytorch/sequential_base.py"
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)
FAILURES = [
    "test_slirec_cross_batch_ranking_groups[1-1]",
    "test_slirec_cross_batch_ranking_groups[1-2]",
    "test_slirec_cross_batch_ranking_groups[1-4]",
    "test_slirec_cross_batch_ranking_groups[3-1]",
    "test_slirec_cross_batch_ranking_groups[3-4]",
    "test_slirec_cross_batch_ranking_groups[8-2]",
    "test_slirec_cross_batch_ranking_groups[8-4]",
    "test_slirec_cross_batch_ranking_groups[16-2]",
    "test_slirec_cross_batch_ranking_groups[16-4]",
    "test_slirec_cross_batch_ranking_groups[32-4]",
    "test_slirec_cross_batch_validation_during_fit[8]",
    "test_slirec_cross_batch_validation_during_fit[16]",
    "test_slirec_cross_batch_groups_after_history_filter"
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
results.append(check("new-regressions-before", "cross_batch", 9, FAILURES))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", None, 34, []))
for path in [TEST, SOURCE]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", TEST, SOURCE], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [TEST, SOURCE]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", TEST, SOURCE], check=True)
(OUT / "summary.json").write_text(json.dumps({"base":BASE,"head":HEAD,"results":results},indent=2)+"\n")
print("Baseline, thirteen expected failures, exact improvement and formatting verified.", flush=True)
