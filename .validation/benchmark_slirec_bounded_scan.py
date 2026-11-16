"""Compare full and bounded Time4LSTM scans, including actual file evaluation."""
import importlib
import importlib.util
import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from recommenders.models.deeprec.models.sequential.pytorch.rnn_cell import (
    Time4LSTMCell,
    time4lstm_scan,
)

torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location(
    "slirec_tests", "tests/unit/recommenders/models/test_sli_rec.py"
)
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
SOURCE = "recommenders/models/deeprec/models/sequential/pytorch/rnn_cell.py"
baseline_namespace = {}
exec(compile(subprocess.check_output(["git", "show", BASE + ":" + SOURCE]),
             "unchanged_staging_rnn_cell.py", "exec"), baseline_namespace)
full_scan = baseline_namespace["time4lstm_scan"]
module = importlib.import_module(
    "recommenders.models.deeprec.models.sequential.pytorch.sli_rec"
)


def compare(functions):
    samples = {name: [] for name in functions}
    for _ in range(5):
        for function in functions.values():
            function()
    for index in range(31):
        order = list(functions)
        if index % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter_ns()
            functions[name]()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "median_ms": medians,
        "speedup": medians["full"] / medians["bounded"],
        "samples_ms": samples,
    }


result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "threads": torch.get_num_threads(),
    "warmups": 5,
    "samples": 31,
    "standalone": [],
}
for length in [1, 7, 50]:
    torch.manual_seed(42)
    cell = Time4LSTMCell(32, 40)
    inputs = torch.randn(32, 50, 34, requires_grad=True)
    lengths = torch.full((32,), length, dtype=torch.long)
    with torch.no_grad():
        before = full_scan(cell, inputs, lengths)
        after = time4lstm_scan(cell, inputs, lengths)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    case = {"batch": 32, "padded_length": 50, "valid_length": length}

    def forward(scan):
        with torch.no_grad():
            scan(cell, inputs, lengths)

    def backward(scan):
        cell.zero_grad(set_to_none=True)
        inputs.grad = None
        scan(cell, inputs, lengths).square().mean().backward()

    case["forward"] = compare({
        "full": lambda: forward(full_scan),
        "bounded": lambda: forward(time4lstm_scan),
    })
    case["forward_backward"] = compare({
        "full": lambda: backward(full_scan),
        "bounded": lambda: backward(time4lstm_scan),
    })
    result["standalone"].append(case)
    print(json.dumps({"standalone": case}), flush=True)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    paths = tests.synthetic_slirec.__wrapped__(SimpleNamespace(mktemp=lambda _: root))
    model = tests._build_model(paths, init_value=0.2)
    filename = str(root / "test_data")

    def evaluate(scan):
        module.time4lstm_scan = scan
        try:
            return model.run_eval(filename, num_ngs=4, batch_size=100)
        finally:
            module.time4lstm_scan = time4lstm_scan

    old_metrics, new_metrics = evaluate(full_scan), evaluate(time4lstm_scan)
    assert old_metrics == new_metrics
    evaluation = compare({
        "full": lambda: evaluate(full_scan),
        "bounded": lambda: evaluate(time4lstm_scan),
    })
    evaluation.update({"examples": 40, "padded_length": 50,
                       "maximum_valid_length": 7, "metrics": new_metrics})
    result["file_evaluation"] = evaluation
    print(json.dumps({"file_evaluation": evaluation}), flush=True)
output = Path(__file__).with_suffix(".json")
output.write_text(json.dumps(result, indent=2) + "\n")
