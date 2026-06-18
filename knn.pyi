from typing import List, Optional
from numpy import ndarray

class Metric:
    Euclidean: "Metric"
    Manhattan: "Metric"
    Cosine: "Metric"

class Weighting:
    Uniform: "Weighting"
    InverseDistance: "Weighting"
    SmoothedInverse: "Weighting"
    Gaussian: "Weighting"

class Algorithm:
    BruteForce: "Algorithm"
    KdTree: "Algorithm"

class KnnClassifier:
    def __init__(
        self,
        k: int,
        metric: Optional[Metric] = ...,
        weighting: Optional[Weighting] = ...,
        algorithm: Optional[Algorithm] = ...,
    ) -> None: ...
    def fit(self, x: ndarray, y: ndarray) -> None: ...
    def predict(self, x: ndarray) -> List[int]: ...

