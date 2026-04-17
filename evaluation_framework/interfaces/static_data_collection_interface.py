from abc import abstractmethod, ABC

import torch


class StaticDataCollectionInterface(ABC):
    @abstractmethod
    def get_data(self) -> dict[str: torch.Tensor]:
        pass
