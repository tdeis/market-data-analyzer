from dataclasses import dataclass
import pandas as pd


@dataclass
class ExperimentResult:
    strategy: str
    returns: pd.Series
    metrics: dict
    metadata: dict
