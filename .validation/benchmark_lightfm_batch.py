"""Compare complete native LightFM evaluation helpers with identical inputs."""
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import lightfm
import numpy as np
import pandas as pd
import scipy
from lightfm import LightFM
from scipy import sparse
from recommenders.models.lightfm.lightfm_utils import prepare_all_predictions

BASE = "6fb74f88b2a271b917be462a2a2a59e100b929da"
SOURCE = "recommenders/models/lightfm/lightfm_utils.py"
assert Path(prepare_all_predictions.__code__.co_filename).resolve() == Path(SOURCE).resolve()
baseline_scope = {}
exec(compile(subprocess.check_output(["git", "show", BASE + ":" + SOURCE]),
             "baseline_lightfm_utils.py", "exec"), baseline_scope)
baseline = baseline_scope["prepare_all_predictions"]
records = []
for n_users, n_items in [(40, 40), (100, 150)]:
    for with_features in [False, True]:
        rng = np.random.RandomState(17)
        interactions = sparse.coo_matrix((rng.rand(n_users, n_items) < 0.08).astype(np.float32))
        users = ["user-" + str(i) for i in range(n_users)]
        items = ["item-" + str(i) for i in range(n_items)]
        data = pd.DataFrame({"userID": [users[i % n_users] for i in range(n_items)],
                             "itemID": items})
        user_features = sparse.csr_matrix(rng.rand(n_users, 8).astype(np.float32)) if with_features else None
        item_features = sparse.csr_matrix(rng.rand(n_items, 12).astype(np.float32)) if with_features else None
        model = LightFM(no_components=8, loss="warp", random_state=7).fit(
            interactions, user_features=user_features, item_features=item_features,
            epochs=2, num_threads=1)
        args = (data, dict(zip(users, range(n_users))), dict(zip(items, range(n_items))),
                interactions, model, 1)
        kwargs = dict(user_features=user_features, item_features=item_features)
        expected = baseline(*args, **kwargs)
        actual = prepare_all_predictions(*args, **kwargs)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        timings = {"before": [], "after": []}
        for _ in range(5):
            for label, function in [("before", baseline), ("after", prepare_all_predictions)]:
                started = time.perf_counter()
                output = function(*args, **kwargs)
                timings[label].append(time.perf_counter() - started)
                pd.testing.assert_frame_equal(output, expected, check_exact=True)
        before = statistics.median(timings["before"])
        after = statistics.median(timings["after"])
        record = dict(users=n_users, items=n_items, optional_features=with_features,
                      retained_pairs=len(actual), before_median_seconds=before,
                      after_median_seconds=after, speedup=before / after,
                      samples=timings, exact_dataframe_parity=True,
                      score_sha256=hashlib.sha256(actual.prediction.to_numpy().tobytes()).hexdigest())
        records.append(record)
        print(json.dumps(record), flush=True)

report = dict(base=BASE, python=platform.python_version(), lightfm=lightfm.__version__,
              numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
              threads=1, repeats=5, measured_scope="complete prepare_all_predictions call",
              records=records)
out = Path(".validation/results")
out.mkdir(parents=True, exist_ok=True)
(out / "lightfm-batch-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
