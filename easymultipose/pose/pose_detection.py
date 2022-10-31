from abc import ABC, abstractmethod
from typing import List

from easymultipose.pose.detection import Detection


class PoseDetection(ABC):

    @abstractmethod
    def detect(self, image, camera) -> List[Detection]:
        pass
