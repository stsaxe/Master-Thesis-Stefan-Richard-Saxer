from abc import abstractmethod, ABC


class RegistryInterface(ABC):

    @abstractmethod
    def _get_registry_name(self) -> str:
        pass
    @abstractmethod
    def _get_registry_key(self) -> int | None:
        pass