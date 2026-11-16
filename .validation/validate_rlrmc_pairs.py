"""Validate sparse RLRMC pair prediction against unchanged staging."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "f74abf8dffada2f9f4b70286fd043aba4fd12b04"
SOURCE = "recommenders/models/rlrmc/RLRMCalgorithm.py"
TEST = "tests/unit/recommenders/models/test_rlrmc.py"
CONTROL = "tests/unit/recommenders/models/test_geoimc.py"
FAILURES = ['test_rlrmc_predict_after_native_fit[False-False]', 'test_rlrmc_predict_after_native_fit[False-True]', 'test_rlrmc_predict_after_native_fit[True-True]', 'test_rlrmc_predict_empty[False]', 'test_rlrmc_predict_independent_of_batching[False]', 'test_rlrmc_predict_pair_scores[interleaved-repeats-False--2.5]', 'test_rlrmc_predict_pair_scores[interleaved-repeats-False-0.0]', 'test_rlrmc_predict_pair_scores[interleaved-repeats-False-3.25]', 'test_rlrmc_predict_pair_scores[interleaved-repeats-True--2.5]', 'test_rlrmc_predict_pair_scores[interleaved-repeats-True-3.25]', 'test_rlrmc_predict_pair_scores[one-repeated-pair-False--2.5]', 'test_rlrmc_predict_pair_scores[one-repeated-pair-False-0.0]', 'test_rlrmc_predict_pair_scores[one-repeated-pair-False-3.25]', 'test_rlrmc_predict_pair_scores[one-repeated-pair-True--2.5]', 'test_rlrmc_predict_pair_scores[one-repeated-pair-True-3.25]', 'test_rlrmc_predict_pair_scores[unsorted-unique-True--2.5]', 'test_rlrmc_predict_pair_scores[unsorted-unique-True-3.25]']
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)

def content(revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path])

def restore(revision, path):
    Path(path).write_bytes(content(revision, path))

def check(label, paths, passed, failures):
    xml = OUT / (label + ".xml")
    result = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--junitxml=" + str(xml)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    actual = sorted(c.attrib["name"] for c in cases if c.find("failure") is not None)
    assert actual == sorted(failures), (label, actual)
    assert not any(c.find("error") is not None or c.find("skipped") is not None for c in cases)
    assert len(cases) - len(actual) == passed, (label, len(cases))
    assert result.returncode == (1 if failures else 0)
    return dict(name=label, passed=passed, failures=actual)

subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, HEAD], check=True)
restore(BASE, SOURCE)
results = [check("unchanged-experimental-controls", [CONTROL], 12, [])]
restore(HEAD, TEST)
results.append(check("new-regressions-before", [TEST], 11, FAILURES))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", [TEST, CONTROL], 40, []))
for path in [SOURCE, TEST]:
    assert Path(path).read_bytes() == content(HEAD, path)
assert ("    - " + TEST + "\n").encode() in content(HEAD, "tests/test_groups.yml")
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, TEST], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
restore(BASE, SOURCE)
Path(TEST).unlink()
subprocess.run(["git", "diff", "--exit-code", BASE, "--", SOURCE, TEST], check=True)
(OUT / "summary.json").write_text(json.dumps(dict(base=BASE, head=HEAD, results=results), indent=2)+"\n")
print("Native Pymanopt/Numba baseline, seventeen failures and forty passing tests verified.", flush=True)
