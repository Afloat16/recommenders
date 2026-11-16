"""Verify NewsRec attention against physically removed padding."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "b890261b4c6df7b10e7e9af511b96e247fe8bf5e"
SOURCE = "recommenders/models/newsrec/models/layers.py"
TESTS = ["tests/unit/recommenders/models/test_newsrec_model.py"]
FAILURES = [f"test_additive_attention_masked_shift_matches_removed_positions[{scale}-dtype{dtype}]"
            for scale in ["10.0", "100.0"] for dtype in [0, 1]] + [
                f"test_lstur_masked_shift_matches_unpadded_training[{combine}-dtype{dtype}]"
                for combine in ["ini", "con"] for dtype in [0, 1]]
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
results = [check("unchanged-newsrec-model-suite", 38, [])]
for path in TESTS:
    restore(HEAD, path)
results.append(check("masked-shift-before", 4, FAILURES,
                     "masked_shift"))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", 50, []))
for path in [SOURCE, *TESTS]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, *TESTS], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [SOURCE, *TESTS]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", SOURCE, *TESTS], check=True)
(OUT / "summary.json").write_text(json.dumps(dict(base=BASE, head=HEAD, results=results), indent=2) + "\n")
print("Masked outputs, gradients and native LSTUR training verified.", flush=True)
