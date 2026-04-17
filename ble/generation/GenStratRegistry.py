from ble.walking.Path import Path

class GenStratRegistry:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(GenStratRegistry, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        self._entries: dict[Path, type] = dict()

    def register(self, path_string: str):
        def deco(cls):
            path = Path()
            path.from_string(path_string)

            for p in self._entries.keys():
                if path.matches(p) or p.matches(path):
                    raise KeyError(f"Key {p} already exists")

            self._entries[path] = cls
            return cls

        return deco

    def create(self, ref: Path, **kwargs):
        for p in self._entries.keys():
            if ref.matches(p) or p.matches(ref):
                return self._entries[p](**kwargs)

        raise KeyError(f"Invalid Key, does not exist")

    def __getitem__(self, path: Path) -> type | None:
        if isinstance(path, Path):
            for p in self._entries.keys():
                if path.matches(p) or p.matches(path):
                    return self._entries[p]

            raise KeyError(f"Invalid Key, does not exist")

        else:
            raise KeyError(f"Invalid Key")




