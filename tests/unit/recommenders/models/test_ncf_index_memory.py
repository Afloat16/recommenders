# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import gc
import tracemalloc

from recommenders.models.ncf.dataset import DataFile


def _write_ratings(path, count):
    with path.open("w") as stream:
        stream.write("userID,itemID,rating\n")
        for i in range(count):
            stream.write(f"{i // (count // 10) + 1},{1000 + i % 20},5\n")


def _index_peak(path, count):
    _write_ratings(path, count)
    gc.collect()
    tracemalloc.start()
    try:
        data = DataFile(path, "userID", "itemID", "rating")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert data.data_len == count
    assert data.user2id == {i: i - 1 for i in range(1, 11)}
    assert data.item2id == {i: i - 1000 for i in range(1000, 1020)}
    return peak


def test_file_index_memory_depends_on_entities_not_interaction_count(tmp_path):
    small_peak = _index_peak(tmp_path / "small.csv", 5000)
    large_peak = _index_peak(tmp_path / "large.csv", 50000)

    # Allow allocator and file-buffer variation while detecting retained records.
    assert large_peak < 2 * small_peak + 128 * 1024
