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
