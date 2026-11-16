"""Verify native LightFM similarity results against both base and exact head."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "a85bcc5456aba5db8e3616a3b4c164bea7048f74"
SOURCE = "recommenders/models/lightfm/lightfm_utils.py"
TEST = "tests/unit/recommenders/models/test_lightfm_utils.py"
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)


def restore(revision, path):
    data = subprocess.check_output(["git", "show", revision + ":" + path])
    Path(path).write_bytes(data)


def check(label, passed, failures, skipped=0, select=None):
    xml = OUT / (label + ".xml")
    args = [sys.executable, "-m", "pytest", TEST, "-q", "--junitxml=" + str(xml)]
    if select:
        args += ["-k", select]
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    actual = sorted(x.attrib["name"] for x in cases if x.find("failure") is not None)
    actual_skips = [x.attrib["name"] for x in cases if x.find("skipped") is not None]
    assert actual == sorted(failures), (label, actual)
    assert not any(x.find("error") is not None for x in cases)
    assert len(actual_skips) == skipped, (label, actual_skips)
    if skipped:
        assert actual_skips == ["test_fitting"]
    assert len(cases) - len(actual) - skipped == passed, (label, len(cases))
    assert result.returncode == (1 if failures else 0)
    return dict(label=label, passed=passed, failures=actual, skipped=actual_skips)


subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, HEAD], check=True)
for path in [SOURCE, TEST]:
    restore(BASE, path)
results = [check("unchanged-lightfm-suite", 4, [], skipped=1)]
restore(HEAD, TEST)
failures = [
    f"test_similar_entities_exclude_query_with_ties[{count}-{kind}-{entity}]"
    for count in [1, 3, 5]
    for kind in ["shared", "zero"]
    for entity in ["users", "items"]
]
results.append(check("query-exclusion-before", 12, failures, select="exclude_query"))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", 28, [], skipped=1))
for path in [SOURCE, TEST]:
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", HEAD + ":" + path])
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, TEST], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [SOURCE, TEST]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", SOURCE, TEST], check=True)
(OUT / "summary.json").write_text(json.dumps(dict(base=BASE, head=HEAD, results=results), indent=2) + "\n")
print("Native shared, zero and distinct feature similarity checks verified.", flush=True)
