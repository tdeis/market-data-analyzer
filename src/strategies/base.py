from abc import ABC, abstractmethod


class BaseStrategy(ABC):

    @abstractmethod
    def generate_weights(self, returns, cov):
        pass
