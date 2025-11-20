__all__ = ["Disturbance"]

from typing import List

class Disturbance:
    def __init__(self, estimate_dist: bool = False, estimated_vector_length: int = 0):
        self.estimate_dist = estimate_dist
        self.estimated_vector_length = estimated_vector_length
    