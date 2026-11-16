"""Validate the bounded recurrent scan against unchanged staging."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "747087a8b6cef54a0bdc8545bc3d3bd31148f3c5"
TEST = "tests/unit/recommenders/models/test_sli_rec.py"
SOURCE = "recommenders/models/deeprec/models/sequential/pytorch/rnn_cell.py"
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)
FAILURES = [
    "test_time4lstm_bounded_scan_preserves_outputs_and_gradients[lengths0-1-float32]",
    "test_time4lstm_bounded_scan_preserves_outputs_and_gradients[lengths0-1-float64]",
    "test_time4lstm_bounded_scan_preserves_outputs_and_gradients[lengths1-4-float32]",
    "test_time4lstm_bounded_scan_preserves_outputs_and_gradients[lengths1-4-float64]",
    "test_time4lstm_bounded_scan_preserves_outputs_and_gradients[lengths2-1-float32]",
    "test_time4lstm_bounded_scan_preserves_outputs_and_gradients[lengths2-1-float64]",
    "test_slirec_bounded_scan_preserves_optimizer_step[False-float32]",
    "test_slirec_bounded_scan_preserves_optimizer_step[False-float64]",
    "test_slirec_bounded_scan_preserves_optimizer_step[True-float32]",
    "test_slirec_bounded_scan_preserves_optimizer_step[True-float64]",
    "test_slirec_bounded_scan_preserves_file_evaluation"
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
results.append(check("new-regressions-before", "bounded_scan", 6, FAILURES))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", None, 29, []))
for path in [TEST, SOURCE]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", TEST, SOURCE], check=True)
subprocess.run(["git", "diff", "--check"], check=True)

import torch
if torch.__version__.startswith("2.5.1"):
    with (OUT / "benchmark.log").open("w") as log:
        subprocess.run([sys.executable, ".validation/benchmark_slirec_bounded_scan.py"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(Path(".validation/benchmark_slirec_bounded_scan.json").read_text())
    (OUT / "benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Full file evaluation:", data["file_evaluation"]["median_ms"],
          "speedup:", data["file_evaluation"]["speedup"])

for path in [TEST, SOURCE]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", TEST, SOURCE], check=True)
(OUT / "summary.json").write_text(json.dumps({"base":BASE,"head":HEAD,"results":results},indent=2)+"\n")
print("Baseline, eleven expected failures, exact improvement and formatting verified.", flush=True)
