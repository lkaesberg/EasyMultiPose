from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    name: str
    pose: np.array
    likelihood: float
