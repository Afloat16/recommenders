"""Reproducible CPU timing of the KCNN projection change and complete DKN eval."""
import argparse
import json
import statistics
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recommenders.models.deeprec.models.pytorch.dkn import DKN, KCNN


def full_table_forward(model, words, entities):
    word_table, *knowledge_tables = model.embedding_tables()
    channels = [F.embedding(words, word_table)] + [
        F.embedding(entities, table) for table in knowledge_tables
    ]
    x = torch.cat(channels, dim=-1).transpose(1, 2)
    return torch.cat([F.relu(conv(x)).amax(dim=-1) for conv in model.convs], dim=1)


def measure(run, repeats):
    for _ in range(5):
        run(0)
        run(1)
    samples = [[], []]
    for repeat in range(repeats):
        for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            run(which)
            samples[which].append((time.perf_counter() - start) * 1000)
    before, after = map(statistics.median, samples)
    return {
        "before_ms": before,
        "after_ms": after,
        "speedup": before / after,
        "samples_before_ms": samples[0],
        "samples_after_ms": samples[1],
    }


def benchmark_kcnn():
    results = []
    for vocabulary, batch, length in (
        (10000, 4, 10), (50000, 32, 20), (100000, 64, 50), (64, 64, 20)
    ):
        torch.manual_seed(4)
        rng = np.random.default_rng(5)
        model = KCNN(
            rng.normal(size=(1000, 64)).astype("float32"),
            rng.normal(size=(vocabulary, 48)).astype("float32"),
            rng.normal(size=(vocabulary, 48)).astype("float32"),
            [1, 2, 3], 8, "uniform", 0.1,
        )
        words = torch.randint(0, 1000, (batch, length))
        entities = torch.randint(0, vocabulary, words.shape)
        forwards = (full_table_forward, KCNN.forward)
        with torch.no_grad():
            expected = forwards[0](model, words, entities)
            actual = forwards[1](model, words, entities)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            output_error = (actual - expected).abs().max().item()
        for backward in (False, True):
            def run(which):
                model.forward = types.MethodType(forwards[which], model)
                model.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(backward):
                    output = model(words, entities)
                    if backward:
                        output.square().mean().backward()
            row = {
                "scope": "KCNN forward and backward" if backward else "KCNN forward",
                "vocabulary": vocabulary,
                "batch": batch,
                "length": length,
                "maximum_output_error": output_error,
                **measure(run, 31),
            }
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.startswith("samples")}), flush=True)
    return results


def benchmark_dkn(directory):
    rng = np.random.default_rng(42)
    news_count, word_count, entity_count = 500, 1000, 50000
    doc_size, user_count, sample_count = 20, 32, 512
    for name, shape in (
        ("word", (word_count, 64)),
        ("entity", (entity_count, 48)),
        ("context", (entity_count, 48)),
    ):
        np.save(directory / (name + ".npy"), rng.normal(scale=0.1, size=shape).astype("float32"))
    with (directory / "news.txt").open("w") as stream:
        for article in range(1, news_count + 1):
            words = ",".join(map(str, rng.integers(1, word_count, doc_size)))
            entities = ",".join(map(str, rng.integers(0, entity_count, doc_size)))
            stream.write(f"N{article} {words} {entities}\n")
    with (directory / "history.txt").open("w") as stream:
        for user in range(user_count):
            history = ",".join(f"N{x}" for x in rng.choice(np.arange(1, news_count + 1), 20, replace=False))
            stream.write(f"U{user} {history}\n")
    with (directory / "eval.txt").open("w") as stream:
        for sample in range(sample_count):
            stream.write(f"{sample % 2} U{sample % user_count} N{rng.integers(1, news_count + 1)}%{sample // 4}\n")
    model = DKN(
        str(directory / "news.txt"), str(directory / "history.txt"),
        str(directory / "word.npy"), str(directory / "entity.npy"),
        str(directory / "context.npy"), history_size=20,
        filter_sizes=[1, 2, 3], num_filters=8, attention_layer_size=32,
        layer_sizes=[32], enable_BN=False, seed=42,
    ).to("cpu")
    forwards = (full_table_forward, KCNN.forward)
    def run(which):
        model.kcnn.forward = types.MethodType(forwards[which], model.kcnn)
        return model.run_eval(str(directory / "eval.txt"), batch_size=32)
    expected, actual = run(0), run(1)
    assert expected == actual, (expected, actual)
    row = {
        "scope": "complete DKN run_eval, including article cache, scoring and metrics",
        "news_articles": news_count, "evaluation_rows": sample_count,
        "vocabulary": entity_count, "batch": 32, "length": doc_size,
        "metrics_before": expected, "metrics_after": actual,
        **measure(run, 15),
    }
    print(json.dumps({k: v for k, v in row.items() if not k.startswith("samples")}), flush=True)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = benchmark_kcnn()
    with tempfile.TemporaryDirectory() as directory:
        rows.append(benchmark_dkn(Path(directory)))
    result = {"torch": torch.__version__, "threads": torch.get_num_threads(), "results": rows}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
