# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from recommenders.models.geoimc.geoimc_data import ML_100K


def _write_item_file(path, video_dates, urls):
    genres = np.zeros((3, 19), dtype=int)
    genres[0, 1] = 1
    genres[1, 2] = 1
    genres[2, [1, 2]] = 1
    rows = []
    for i in range(3):
        fields = [i + 1, "Same title", "01-Jan-2000", video_dates[i], urls[i]]
        rows.append("|".join(map(str, [*fields, *genres[i]])))
    path.write_text("\n".join(rows) + "\n", encoding="ISO-8859-1")
    return genres


@pytest.mark.parametrize("with_video_dates", [False, True])
@pytest.mark.parametrize("missing_url", [False, True])
def test_movie_features_encode_each_rows_values(
    tmp_path, with_video_dates, missing_url
):
    video_dates = ["", "01-Jan-2001", "01-Jan-2002"] if with_video_dates else [""] * 3
    urls = [
        "https://imdb.example/a",
        "https://imdb.example/b",
        "https://imdb.example/a",
    ]
    if missing_url:
        urls[2] = ""
    path = tmp_path / "u.item"
    genres = _write_item_file(path, video_dates, urls)

    features = ML_100K()._load_item_features(path)

    categories = [np.ones((3, 2))]
    if with_video_dates:
        categories.append(np.array([[0, 0], [1, 0], [0, 1]]))
    categories.append(np.array([[1, 0], [0, 1], [0 if missing_url else 1, 0]]))
    expected = np.concatenate([*categories, genres], axis=1)
    assert np.issubdtype(features.dtype, np.floating)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, expected)


@pytest.mark.parametrize("normalize", [False, True])
def test_movie_features_survive_dataset_normalization_and_pca(tmp_path, normalize):
    _write_item_file(
        tmp_path / "u.item",
        ["", "", ""],
        ["https://imdb.example/a", "https://imdb.example/b", "https://imdb.example/c"],
    )
    (tmp_path / "u.user").write_text(
        "1|24|M|student|10001\n2|30|F|engineer|20002\n3|41|M|writer|30003\n"
    )
    (tmp_path / "u1.base").write_text("1\t1\t5\t1\n2\t2\t3\t2\n3\t3\t4\t3\n")
    (tmp_path / "u1.test").write_text("1\t2\t2\t4\n")
    dataset = ML_100K(normalize=normalize, features_dim=2)

    dataset.load_data(tmp_path)

    assert dataset.entities[1].shape == (3, 2)
    assert np.isfinite(dataset.entities[1]).all()
    assert dataset.training_data.get_entity("col") is dataset.entities[1]
    assert dataset.test_data.get_entity("col") is dataset.entities[1]
    np.testing.assert_allclose(dataset.entities[1].mean(axis=0), 0, atol=1e-12)
