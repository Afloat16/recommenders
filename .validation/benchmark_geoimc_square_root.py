"""Compare complete GeoIMC inference with identical fixed inputs."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info, threadpool_limits

from recommenders.models.geoimc.geoimc_data import DataPtr

SOURCE = "recommenders/models/geoimc/geoimc_predict.py"
BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"


def load(source, name):
    module = types.ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=".")
    parser.add_argument("--head")
    parser.add_argument("--candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def git_source(revision):
        return subprocess.check_output(
            ["git", "-C", args.repository, "show", revision + ":" + SOURCE], text=True
        )

    original_source = git_source(BASE)
    candidate_source = (
        Path(args.candidate).read_text() if args.candidate else git_source(args.head)
    )
    before = load(original_source, "geoimc_before")
    after = load(candidate_source, "geoimc_after")
    rng = np.random.default_rng(20261116)
    results = []
    with threadpool_limits(limits=1):
        for rank, users, items, features in [(8, 943, 1682, 128), (32, 1, 256, 128), (128, 1, 256, 256)]:
            x = rng.normal(size=(users, features))
            z = rng.normal(size=(items, features))
            u = np.linalg.qr(rng.normal(size=(features, rank)))[0]
            v = np.linalg.qr(rng.normal(size=(features, rank)))[0]
            q = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
            b = (q * np.linspace(0.1, 2.0, rank)) @ q.T
            data = DataPtr(csr_matrix((users, items)), [x, z])
            weights = [u, b, v]
            old = before.Inferer()
            new = after.Inferer()
            expected = old.infer(data, weights)
            actual = new.infer(data, weights)
            assert np.array_equal(expected, actual)

            calls = {}
            for label, module, inferer in [("before", before, old), ("after", after, new)]:
                function = module.sqrtm
                count = [0]

                def counted(matrix):
                    count[0] += 1
                    return function(matrix)

                module.sqrtm = counted
                assert np.array_equal(inferer.infer(data, weights), expected)
                module.sqrtm = function
                calls[label] = count[0]
            assert calls == {"before": 2, "after": 1}

            for _ in range(5):
                old.infer(data, weights)
                new.infer(data, weights)
            times = {"before": [], "after": []}
            for repeat in range(31):
                order = [("before", old), ("after", new)]
                if repeat % 2:
                    order.reverse()
                for label, inferer in order:
                    started = time.perf_counter()
                    result = inferer.infer(data, weights)
                    times[label].append((time.perf_counter() - started) * 1000)
                    assert np.array_equal(result, expected)
            medians = {name: float(np.median(values)) for name, values in times.items()}
            results.append(
                dict(rank=rank, users=users, items=items, features=features,
                     median_ms=medians, speedup=medians["before"] / medians["after"],
                     sqrt_calls=calls, full_output_equal=True,
                     output_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                     samples_ms=times)
            )
        environment = dict(python=platform.python_version(), numpy=np.__version__,
                           scipy=scipy.__version__, pools=threadpool_info())
    report = dict(base=BASE, head=args.head, environment=environment,
                  original_source_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
                  candidate_source_sha256=hashlib.sha256(candidate_source.encode()).hexdigest(),
                  warmup=5, measured=31, alternating_order=True, results=results)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in item.items() if k != "samples_ms"} for item in results], indent=2))


if __name__ == "__main__":
    main()
