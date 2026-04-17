from typing import Callable, TYPE_CHECKING


class ParsePolicyRegistry:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(ParsePolicyRegistry, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        self.names: dict[str, type] = dict()

    def register(self, name: str, *args) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            assert name not in self.names.keys(), "name already exists"
            self.names[name] = cls(*args)
            return cls

        return decorator

    def __getitem__(self, item) -> type:
        if isinstance(item, str):
            return self.names[item]
        else:
            raise KeyError(f"Invalid Advertising Registry Key {item}, must be int or str")

PARSE_POLICY_REGISTRY = ParsePolicyRegistry()