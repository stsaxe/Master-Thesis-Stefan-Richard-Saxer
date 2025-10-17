from __future__ import annotations
import pandas as pd

from tgf.flags.abstract_flag import AbstractFlag
from tgf.flags.base_flag import BaseFlag
from tgf.interfaces.executor_interface import ExecutorInterface
from tgf.tasks.abstract_task import AbstractTask
from tgf.tasks.simple_executor import SimpleExecutor


class Task(AbstractTask):
    def __init__(self, name: str, priority: int = None, flags: list[AbstractFlag] | AbstractFlag = BaseFlag(),
                 inplace: bool = True, executor: ExecutorInterface = SimpleExecutor()):
        super().__init__(name, priority, flags, inplace)
        self.__executor = executor

    def getExecutor(self) -> ExecutorInterface:
        return self.__executor

    def process(self, dataToProcess: pd.DataFrame, flag: AbstractFlag) -> pd.DataFrame:
        for parent in self.getFlags():
            if flag.contains(parent):
                return self.execute(dataToProcess)

        return dataToProcess

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        if self.isInplace():
            return self.__executor.execute(dataToProcess)
        else:
            self.__executor.execute(dataToProcess.copy(deep=True))
            return dataToProcess

    def copy(self) -> AbstractTask:
        return Task(self.getName(), self.getPriority(), self.getFlags(), self.isInplace(), executor=self.__executor)
