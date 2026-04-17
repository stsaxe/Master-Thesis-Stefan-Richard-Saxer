from abc import abstractmethod, ABC
from typing import Callable, override

from ble.utils.HelperMethods import HelperMethods


class ComponentRegistry(HelperMethods, ABC):
    def __init__(self):
        self.registry: dict[int, type] = dict()
        self.reverse_registry: dict[type, int] = dict()
        self.names: dict[str, type] = dict()

    @abstractmethod
    def register(self, key: int | None, name: str) -> Callable[[type], type]:
        pass

    def get_names(self) -> dict[str, type]:
        return self.names

    def get_registry(self) -> dict[int, type]:
        return self.registry

    def __getitem__(self, item) -> type:
        if isinstance(item, int):
            return self.registry[item]
        elif isinstance(item, str):
            return self.names[item]
        else:
            raise KeyError(f"Invalid Advertising Registry Key {item}, must be int or str")

    def __len__(self) -> int:
        return len(self.registry.keys())


class AdvertisingRegistry(ComponentRegistry):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(ComponentRegistry, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        ComponentRegistry.__init__(self)
        self.occurrences: dict[int, dict[str, str]] = dict()
        self.occurrences_per_context: dict[str, dict[int, int]] = dict()

    def requirements(self, key: int | None, context: dict[str, str] | None) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            if key is None and context is not None:
                raise Warning("Context is ignored if Key is None")

            assert isinstance(context, dict), "context must be dict or None"
            for k, v in context.items():
                assert isinstance(k, str), "dict key must be str"
                assert isinstance(v, str), "dict value must be str"
                assert v in ['O', 'X', 'C1', 'C2'], "dict value must be one of ['O', 'X', 'C1', 'C2']"

            if key is not None:
                assert key not in self.occurrences.keys(), "key is already registered"
                self.occurrences[key] = context

                self._register_occurrences_per_context()

            return cls

        return decorator

    @override
    def register(self, key: int | None, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            assert isinstance(key, int) or key is None, "key must be int or None"
            assert isinstance(name, str), "name must be str or None"

            n = HelperMethods.check_valid_string(name)
            k = key

            if k is not None:
                assert 0 <= k <= 255, f'invalid Advertising Registry Key {k}'
                if k in self.registry:
                    raise KeyError(f"Duplicate registry key: {k}")
                self.registry[k] = cls
                self.reverse_registry[cls] = k
            cls.ADVERTISING_REGISTRY_KEY = k

            if n in self.names:
                raise KeyError(f"Duplicate name key: {k}")

            self.names[n] = cls
            cls.ADVERTISING_REGISTRY_NAME = n

            return cls

        return decorator

    def get_occurrences(self, item) -> dict[str, str]:
        return self.occurrences[item]

    @staticmethod
    def _conver_string_to_max_count_int(string: str):
        assert string in ['O', 'X', 'C1', 'C2'], "occurrence must be 'O', 'X', 'C1', 'C2'"
        if string == 'O':
            # de facto infinite
            return 100_000
        elif string == 'X':
            return 0
        else:
            return 1

    def _register_occurrences_per_context(self):
        self.occurrences_per_context: dict[str, dict[int, int]] = dict()
        for context in ['EIR', 'AD', 'SRD', 'ACAD', 'OOB']:
            self.occurrences_per_context[context] = self._get_max_occurrences_for_context(context)

    def _get_max_occurrences_for_context(self, context: str) -> dict[int, int]:
        assert context in ['EIR', 'AD', 'SRD', 'ACAD',
                           'OOB'], "invalid context, must be EIR or AD or SRD or ACAD or OOB"

        context_occurrences: dict[int, int] = dict()
        for adv_type in self.occurrences.keys():
            max_occurrence = self.occurrences[adv_type][context]
            context_occurrences[adv_type] = self._conver_string_to_max_count_int(max_occurrence)

        return context_occurrences

    def get_occurrences_per_context(self, context: str) -> dict[int, int]:
        return self.occurrences_per_context[context]


class PDURegistry(ComponentRegistry):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(ComponentRegistry, cls).__new__(cls)
        return cls.__instance

    @override
    def register(self, key: int | None, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            assert isinstance(key, int) or key is None, "key must be int or None"
            assert isinstance(name, str), "name must be str or None"

            n = HelperMethods.check_valid_string(name)
            k = key

            if k is not None:
                assert 0 <= k <= 15, f'invalid Advertising Registry Key {k}'
                if k in self.registry:
                    raise KeyError(f"Duplicate registry key: {k}")
                self.registry[k] = cls
                self.reverse_registry[cls] = k
            cls.PDU_REGISTRY_KEY = k

            if n in self.names:
                raise KeyError(f"Duplicate name key: {k}")

            self.names[n] = cls
            cls.PDU_REGISTRY_NAME = n

            return cls

        return decorator
