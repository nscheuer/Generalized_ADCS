from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class Subplot(ABC):
    @abstractmethod
    def plot(self, ax: plt.Axes, sim) -> None:
        pass