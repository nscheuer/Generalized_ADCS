from abc import ABC
import matplotlib.pyplot as plt

class Subplot(ABC):
    def plot(self, ax: plt.Axes, sim) -> None:
        pass