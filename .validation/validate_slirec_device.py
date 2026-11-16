"""Validate native device migration and attention-test compatibility."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
HEAD = "4d1b220158d15d93636e565f2298560fe0cb484c"
MASK_HEAD = "9cdeeb972c7909405ca9ff32778ca98baa0441f6"
SOURCE = "recommenders/models/deeprec/models/sequential/pytorch/sequential_base.py"
MASK = "recommenders/models/deeprec/models/sequential/pytorch/sli_rec.py"
TEST = "tests/unit/recommenders/models/test_sli_rec.py"
FAILURES = [
    "test_slirec_device_follows_native_moves[to]",
    "test_slirec_device_follows_native_moves[to_empty]",
    "test_slirec_migrated_batch_runs_native_forward",
    "test_slirec_migrated_regularizer_uses_model_device",
]
CUDA_SKIPS = ["test_slirec_device_cuda_to_cpu[cpu]", "test_slirec_device_cuda_to_cpu[to]"]
HALF_CASES = [f"test_attention_mask_is_representable_in_score_dtype[{n}-float16]" for n in [0, 3, 5]] + ["test_slirec_cpu_autocast_forward_and_training_step[float16]"]
OUT = Path(".validation/results")
OUT.mkdir(parents=True, exist_ok=True)


def content(revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path])


def restore(revision, path):
    Path(path).write_bytes(content(revision, path))


def check(label, passed, failures, skipped, select=None):
    xml = OUT / (label + ".xml")
    args = [sys.executable, "-m", "pytest", TEST, "-q", "--junitxml=" + str(xml)]
    if select:
        args += ["-k", select]
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    actual_failures = sorted(x.attrib["name"] for x in cases if x.find("failure") is not None)
    actual_skips = sorted(x.attrib["name"] for x in cases if x.find("skipped") is not None)
    assert actual_failures == sorted(failures), (label, actual_failures)
    assert actual_skips == sorted(skipped), (label, actual_skips)
    assert not any(x.find("error") is not None for x in cases)
    assert len(cases) - len(actual_failures) - len(actual_skips) == passed, (label, len(cases))
    assert result.returncode == (1 if failures else 0)
    return dict(label=label, passed=passed, failures=actual_failures, skipped=actual_skips)


def half_cpu_supported():
    try:
        torch.nn.functional.linear(torch.ones(2, 2, dtype=torch.float16), torch.ones(2, 2, dtype=torch.float16))
        with torch.autocast("cpu", dtype=torch.float16):
            value = torch.nn.functional.linear(torch.ones(2, 2), torch.ones(2, 2))
        return value.dtype == torch.float16
    except RuntimeError as error:
        if "not implemented for" in str(error) or "AutocastCPU only support" in str(error):
            print("Independent CPU-half capability probe:", str(error), flush=True)
            return False
        raise


assert not torch.cuda.is_available(), "This validation matrix is intentionally CPU-only"
subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", BASE, HEAD, MASK_HEAD], check=True)
for path in [SOURCE, MASK, TEST]:
    restore(BASE, path)
results = [check("unchanged-model-suite", 12, [], [])]
restore(HEAD, TEST)
results.append(check("new-device-cases-before", 1, FAILURES, CUDA_SKIPS, "device or migrated"))
restore(HEAD, SOURCE)
results.append(check("exact-device-head-after", 17, [], CUDA_SKIPS))
for path in [SOURCE, TEST]:
    assert Path(path).read_bytes() == content(HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", SOURCE, TEST], check=True)
restore(BASE, SOURCE)
restore(MASK_HEAD, TEST)
half_supported = half_cpu_supported()
results.append(check("compatible-attention-cases-before", 10,
                     HALF_CASES if half_supported else [], [] if half_supported else HALF_CASES,
                     "representable or autocast"))
restore(MASK_HEAD, MASK)
results.append(check("exact-attention-head-after", 26 if half_supported else 22, [], [] if half_supported else HALF_CASES))
for path in [MASK, TEST]:
    assert Path(path).read_bytes() == content(MASK_HEAD, path)
subprocess.run([sys.executable, "-m", "black", "--check", MASK, TEST], check=True)

# Combine the two independent changes and their actual test additions. Neither
# contribution branch depends on the other, and the full combined suite must run.
restore(HEAD, SOURCE)
device_tests = content(HEAD, TEST).decode().split("\ndef _device_batch(", 1)[1]
Path(TEST).write_text(content(MASK_HEAD, TEST).decode() + "\n\ndef _device_batch(" + device_tests)
results.append(check("combined-device-and-attention", 31 if half_supported else 27, [], CUDA_SKIPS + ([] if half_supported else HALF_CASES)))
for path in [SOURCE, MASK, TEST]:
    restore(BASE, path)
subprocess.run(["git", "diff", "--check"], check=True)
subprocess.run(["git", "diff", "--exit-code", BASE, "--", SOURCE, MASK, TEST], check=True)
(OUT / "summary.json").write_text(json.dumps(dict(base=BASE, head=HEAD, attention_head=MASK_HEAD, results=results), indent=2) + "\n")
print("Native placement, restored CPU predictions/training and independent-change compatibility verified.", flush=True)
