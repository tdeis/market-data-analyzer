class StrategyRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):

        def wrapper(strategy_class):
            cls._registry[name] = strategy_class
            return strategy_class

        return wrapper

    @classmethod
    def get(cls, name):

        if name not in cls._registry:
            raise ValueError(f"Strategy {name} not registered")

        return cls._registry[name]
