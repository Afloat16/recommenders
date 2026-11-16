"""Validate the fusion recency correction against unchanged staging."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "034170008fe29d2c455bf6f7f355dd23f7b0393b"
TEST = "tests/unit/recommenders/models/test_sli_rec.py"
SOURCE = "recommenders/models/deeprec/models/sequential/pytorch/sli_rec.py"
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)
FAILURES = [
    "test_slirec_fusion_uses_latest_observed_time[5-1-0]",
    "test_slirec_fusion_uses_latest_observed_time[5-1-2]",
    "test_slirec_fusion_uses_latest_observed_time[5-2-0]",
    "test_slirec_fusion_uses_latest_observed_time[5-2-2]",
    "test_slirec_fusion_ignores_padding_with_empty_and_mixed_histories",
    "test_slirec_recency_matches_reference_optimizer_step[float32]",
    "test_slirec_recency_matches_reference_optimizer_step[float64]",
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
results.append(check("new-regressions-before", "fusion or recency", 8, FAILURES))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", None, 27, []))
for path in [TEST, SOURCE]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", TEST, SOURCE], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [TEST, SOURCE]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", TEST, SOURCE], check=True)
(OUT / "summary.json").write_text(json.dumps({"base":BASE,"head":HEAD,"results":results},indent=2)+"\n")
print("Baseline, seven expected failures, exact improvement and formatting verified.", flush=True)
