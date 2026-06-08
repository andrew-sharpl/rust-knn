from typing import List, Optional
from numpy import ndarray

class Metric:
    Euclidean: "Metric"
    Manhattan: "Metric"
    Cosine: "Metric"

class KnnClassifier:
    def __init__(self, k: int, metric: Optional[Metric] = ...) -> None: ...
    def fit(self, x: ndarray, y: ndarray) -> None: ...
    def predict(self, x: ndarray) -> List[int]: ...
