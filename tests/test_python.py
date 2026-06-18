import numpy as np
import pytest
import knn


@pytest.fixture
def simple_model():
    model = knn.KnnClassifier(1)
    model.fit(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]]),
        np.array([0, 0, 1]),
    )
    return model


def test_basic_prediction(simple_model):
    predictions = simple_model.predict(np.array([[0.1, 0.0]]))
    assert predictions == [0]


def test_distant_prediction(simple_model):
    predictions = simple_model.predict(np.array([[0.0, 9.0]]))
    assert predictions == [1]


def test_multiple_queries():
    model = knn.KnnClassifier(2)
    model.fit(
        np.array([[0.0], [1.0], [10.0], [11.0]]),
        np.array([0, 0, 1, 1]),
    )
    predictions = model.predict(np.array([[0.4], [10.4]]))
    assert predictions == [0, 1]


def test_k4_ignores_far_points():
    model = knn.KnnClassifier(4)
    model.fit(
        np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [100.0, 100.0],
            [0.0, 10.0],
        ]),
        np.array([0, 0, 0, 0, 1]),
    )
    predictions = model.predict(np.array([[0.05, 0.05]]))
    assert predictions == [0]


@pytest.mark.parametrize("metric", [
    knn.Metric.Euclidean,
    knn.Metric.Manhattan,
    knn.Metric.Cosine,
])
def test_all_metrics_return_prediction(metric):
    model = knn.KnnClassifier(1, metric=metric)
    model.fit(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0, 1]),
    )
    predictions = model.predict(np.array([[0.9, 0.1]]))
    assert len(predictions) == 1


def test_euclidean_matches_sklearn():
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(42)
    X_train = np.random.rand(200, 10)
    y_train = np.random.randint(0, 3, size=200)
    X_test = np.random.rand(50, 10)

    sk_model = KNeighborsClassifier(n_neighbors=3, algorithm="brute", metric="euclidean")
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    rust_model = knn.KnnClassifier(3, metric=knn.Metric.Euclidean)
    rust_model.fit(X_train, y_train)
    rust_preds = rust_model.predict(X_test)

    assert list(sk_preds) == rust_preds


def test_manhattan_matches_sklearn():
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(42)
    X_train = np.random.rand(200, 10)
    y_train = np.random.randint(0, 3, size=200)
    X_test = np.random.rand(50, 10)

    sk_model = KNeighborsClassifier(n_neighbors=3, algorithm="brute", metric="manhattan")
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    rust_model = knn.KnnClassifier(3, metric=knn.Metric.Manhattan)
    rust_model.fit(X_train, y_train)
    rust_preds = rust_model.predict(X_test)

    assert list(sk_preds) == rust_preds


def test_cosine_matches_sklearn():
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(42)
    X_train = np.random.rand(200, 10)
    y_train = np.random.randint(0, 3, size=200)
    X_test = np.random.rand(50, 10)

    sk_model = KNeighborsClassifier(n_neighbors=3, algorithm="brute", metric="cosine")
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    rust_model = knn.KnnClassifier(3, metric=knn.Metric.Cosine)
    rust_model.fit(X_train, y_train)
    rust_preds = rust_model.predict(X_test)

    assert list(sk_preds) == rust_preds


def test_cosine_valid_query():
    model = knn.KnnClassifier(1, metric=knn.Metric.Cosine)
    model.fit(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0, 1]),
    )
    predictions = model.predict(np.array([[0.9, 0.1]]))
    assert predictions == [0]


@pytest.mark.parametrize("weighting", [
    knn.Weighting.Uniform,
    knn.Weighting.InverseDistance,
    knn.Weighting.SmoothedInverse,
    knn.Weighting.Gaussian,
])
def test_all_weightings_return_prediction(weighting):
    model = knn.KnnClassifier(3, weighting=weighting)
    model.fit(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]]),
        np.array([0, 0, 1]),
    )
    predictions = model.predict(np.array([[0.1, 0.0]]))
    assert len(predictions) == 1


def test_inverse_distance_matches_sklearn():
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(42)
    X_train = np.random.rand(200, 10)
    y_train = np.random.randint(0, 3, size=200)
    X_test = np.random.rand(50, 10)

    sk_model = KNeighborsClassifier(n_neighbors=3, algorithm="brute", weights="distance")
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    rust_model = knn.KnnClassifier(3, weighting=knn.Weighting.InverseDistance)
    rust_model.fit(X_train, y_train)
    rust_preds = rust_model.predict(X_test)

    assert list(sk_preds) == rust_preds


def test_inverse_distance_closer_wins():
    model = knn.KnnClassifier(2, weighting=knn.Weighting.InverseDistance)
    model.fit(
        np.array([[0.0], [10.0], [0.1]]),
        np.array([0, 1, 0]),
    )
    predictions = model.predict(np.array([[0.05]]))
    assert predictions == [0]


def test_kdtree_matches_bruteforce():
    np.random.seed(42)
    X_train = np.random.rand(200, 10)
    y_train = np.random.randint(0, 3, size=200)
    X_test = np.random.rand(50, 10)

    brute_model = knn.KnnClassifier(3)
    brute_model.fit(X_train, y_train)
    brute_preds = brute_model.predict(X_test)

    kdtree_model = knn.KnnClassifier(3, algorithm=knn.Algorithm.KdTree)
    kdtree_model.fit(X_train, y_train)
    kdtree_preds = kdtree_model.predict(X_test)

    assert brute_preds == kdtree_preds


def test_kdtree_manhattan_matches_bruteforce():
    np.random.seed(42)
    X_train = np.random.rand(200, 10)
    y_train = np.random.randint(0, 3, size=200)
    X_test = np.random.rand(50, 10)

    brute_model = knn.KnnClassifier(3, metric=knn.Metric.Manhattan)
    brute_model.fit(X_train, y_train)
    brute_preds = brute_model.predict(X_test)

    kdtree_model = knn.KnnClassifier(
        3,
        metric=knn.Metric.Manhattan,
        algorithm=knn.Algorithm.KdTree,
    )
    kdtree_model.fit(X_train, y_train)
    kdtree_preds = kdtree_model.predict(X_test)

    assert brute_preds == kdtree_preds


def test_kdtree_matches_sklearn():
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(42)
    X_train = np.random.rand(200, 10)
    y_train = np.random.randint(0, 3, size=200)
    X_test = np.random.rand(50, 10)

    sk_model = KNeighborsClassifier(n_neighbors=3, algorithm="brute", metric="euclidean")
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    rust_model = knn.KnnClassifier(3, algorithm=knn.Algorithm.KdTree)
    rust_model.fit(X_train, y_train)
    rust_preds = rust_model.predict(X_test)

    assert list(sk_preds) == rust_preds


def test_kdtree_with_cosine_raises_value_error():
    model = knn.KnnClassifier(1, metric=knn.Metric.Cosine, algorithm=knn.Algorithm.KdTree)
    with pytest.raises(ValueError, match="KD-tree pruning is invalid for cosine"):
        model.fit(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0, 1]),
        )


def test_zero_k_raises_value_error():
    with pytest.raises(ValueError, match="k must be positive"):
        knn.KnnClassifier(0)


def test_k_larger_than_training_set_raises_value_error():
    model = knn.KnnClassifier(5)
    with pytest.raises(ValueError, match="cannot be larger than the number of training points"):
        model.fit(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([0, 1, 0]),
        )


def test_label_count_mismatch_raises_value_error():
    model = knn.KnnClassifier(1)
    with pytest.raises(ValueError, match="expected 3 labels"):
        model.fit(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([0, 1]),
        )


def test_predict_before_fit_raises_value_error():
    model = knn.KnnClassifier(1)
    with pytest.raises(ValueError, match="model has not been fit"):
        model.predict(np.array([[0.0, 0.0]]))


def test_dimension_mismatch_raises_value_error():
    model = knn.KnnClassifier(1)
    model.fit(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([0, 1]),
    )
    with pytest.raises(ValueError, match="query dimension .* does not match training dimension"):
        model.predict(np.array([[0.0, 0.0, 0.0]]))


def test_cosine_zero_query_vector_raises_value_error():
    model = knn.KnnClassifier(1, metric=knn.Metric.Cosine)
    model.fit(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0, 1]),
    )
    with pytest.raises(ValueError, match="undefined for the zero vector"):
        model.predict(np.array([[0.0, 0.0]]))
