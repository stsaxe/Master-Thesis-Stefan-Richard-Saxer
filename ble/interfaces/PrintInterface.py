from abc import ABC, abstractmethod


class PrintInterface(ABC):

    def print(self) -> None:
        self._print(ident=0)

    @abstractmethod
    def _print(self, ident: int=0) -> None:
        pass
