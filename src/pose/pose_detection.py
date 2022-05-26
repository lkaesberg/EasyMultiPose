from abc import ABC, abstractmethod


class PoseDetection(ABC):

    @abstractmethod
    def detect(self, image, camera):
        pass
