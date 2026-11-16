"""Validate GeoIMC square-root reuse and full inference equivalence."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "1e0d50e49448c5496d50c2c8bf9cb3654d2a47d1"
SOURCE = "recommenders/models/geoimc/geoimc_predict.py"
TEST = "tests/unit/recommenders/models/test_geoimc.py"
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)

def content(revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path])

def restore(revision, path):
    Path(path).write_bytes(content(revision, path))

def check(label, count):
    xml = OUT / (label + ".xml")
    result = subprocess.run([sys.executable, "-m", "pytest", TEST, "-q", "--junitxml=" + str(xml)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    assert len(cases) == count, (label, len(cases))
    assert not any(x.find("error") is not None or x.find("failure") is not None or x.find("skipped") is not None for x in cases)
    assert result.returncode == 0
    return dict(name=label, passed=count)

subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, HEAD], check=True)
for path in [SOURCE, TEST]:
    restore(BASE, path)
results = [check("unchanged-geoimc", 12)]
restore(HEAD, TEST)
results.append(check("expanded-equivalence-before", 26))
restore(HEAD, SOURCE)
results.append(check("exact-head-after", 26))
for path in [SOURCE, TEST]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, ".validation/benchmark_geoimc_square_root.py",
                "--head", HEAD, "--output", str(OUT / "benchmark.json")], check=True)
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, TEST], check=True)
subprocess.run(["git", "diff", "--check"], check=True)
for path in [SOURCE, TEST]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", SOURCE, TEST], check=True)
(OUT / "summary.json").write_text(json.dumps(dict(base=BASE, head=HEAD, results=results),indent=2)+"\n")
print("GeoIMC functional equivalence and complete inference benchmark verified.", flush=True)
