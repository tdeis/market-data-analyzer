import importlib
import pkgutil
import src.strategies as strategies


def load_all_strategies():

    for _, module_name, _ in pkgutil.iter_modules(strategies.__path__):
        importlib.import_module(f"src.strategies.{module_name}")
