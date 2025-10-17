import pandas as pd
from tgf.interfaces.executor_interface import ExecutorInterface


class SimpleExecutor(ExecutorInterface):
    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        return dataToProcess
