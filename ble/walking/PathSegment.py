import re
from typing import Self, List

from ble.utils.HelperMethods import HelperMethods


class PathSegment(HelperMethods):
    SINGLE_PLACEHOLDER = "*"
    MULTI_PLACEHOLDER = "**"

    def __init__(self, name: str = None, keys: dict[str, str | List[str]] = None):
        self.name: str = None
        if name is not None:
            self._set_name(name)

        self.keys: dict[str, List[str]] = dict()
        self._set_keys(keys)

    def is_placeholder(self) -> bool:
        return self.is_multi_placeholder() or self.is_single_placeholder()

    def is_single_placeholder(self) -> bool:
        return self.name == self.SINGLE_PLACEHOLDER


    def is_multi_placeholder(self) -> bool:
        return self.name == self.MULTI_PLACEHOLDER

    def matches(self, segment: Self) -> bool:

        # the self object should be the scope path, the right object should be a module / component path

        if self.is_placeholder() or segment.is_placeholder():
            return True

        # if the names do not match, return immediately
        if self.name != segment.name:
            return False

        # if there are no slicing keys to compare, return true, as names already match
        if self.keys == dict():
            return True

        for key, value in self.keys.items():
            if key not in segment.keys.keys():
                return False

            if self.SINGLE_PLACEHOLDER in value:
                continue

            compare_value = segment.keys[key]

            if self.SINGLE_PLACEHOLDER in compare_value:
                continue

            # if the cut-set of both lists is empty, no matching values are present
            if len(list(set(value) & set(compare_value))) == 0:
                return False

        return True

    @staticmethod
    def _string_is_placeholder(value: str) -> bool:
        if value == PathSegment.SINGLE_PLACEHOLDER or value == PathSegment.MULTI_PLACEHOLDER:
            return True
        else:
            return False

    def from_string(self, segment: str) -> None:
        if self._string_is_placeholder(segment):
            self._set_name(segment)
            self.keys = dict()

        _SEGMENT_RE = re.compile(r"^(?P<name>[^\[\]]+)(?:\[(?P<keys>[^\]]+)])?$")
        m = _SEGMENT_RE.match(segment)

        assert m, f"invalid selector segment: {segment}"

        name = m.group("name")
        raw_keys = m.group("keys")
        keys = dict()

        if raw_keys:
            for item in raw_keys.split(","):
                k, v = item.split("=", 1)
                k = self._convert_key(k)
                v = self._convert_value(v)

                if k not in keys.keys():
                    keys[k] = [v]
                else:
                    keys[k].append(v)

        self._set_name(self._convert_name(name))
        self._set_keys(keys)



    @staticmethod
    def _convert_name(name: str) -> str:
        HelperMethods.check_valid_string(name)
        assert "." not in name, "name must not contain a dot."
        name = name.strip().lower().replace(" ", "_")

        if PathSegment._string_is_placeholder(name):
            return name
        pattern = re.compile(r'^[A-Za-z0-9*()_\\/-]+$')
        assert re.fullmatch(pattern,
                            name), r"Invalid name: only uppercase letters, lowercase letters, digits, and underscore or / or \ are allowed."

        if PathSegment.SINGLE_PLACEHOLDER in name:
            assert name == PathSegment.SINGLE_PLACEHOLDER or PathSegment.MULTI_PLACEHOLDER, "name cannot contain a * unless it is * or **"
        return name

    @staticmethod
    def _convert_key(key: str) -> str:
        HelperMethods.check_valid_string(key)
        assert "." not in key, "key must not contain a dot."
        key = key.strip().lower().replace(" ", "_")
        pattern = re.compile(r'^[A-Za-z0-9()_\\/-]+$')
        assert re.fullmatch(pattern,
                            key), r"Invalid key: only uppercase letters, lowercase letters, digits, and underscore or / or \ are allowed."

        return key

    @staticmethod
    def _convert_value(value: str) -> str:
        HelperMethods.check_valid_string(value)

        assert "." not in value, "key must not contain a dot."
        value = value.strip().lower().replace(" ", "_")

        pattern = re.compile(r'^[A-Za-z0-9*()_\\/-]+$')
        assert re.fullmatch(pattern,
                            value), r"Invalid value: only uppercase letters, lowercase letters, digits, and underscore or / or \ are allowed."

        if PathSegment.SINGLE_PLACEHOLDER in value:
            assert PathSegment.SINGLE_PLACEHOLDER == value, "value cannot contain * unless it is set to *"

        return value


    def _set_name(self, name: str):
        self.check_valid_string(name, empty_allowed=False)
        self.name = self._convert_name(name)

    def _set_keys(self, keys:  dict[str, str | List[str]]) -> None:
        new_keys = dict()

        if keys is not None and len(keys) > 0:
            assert self.name is not None, "Name cannot be None"
            assert not PathSegment._string_is_placeholder(self.name), "keys cannot be set for raw segments"

            for k, v in keys.items():
                assert isinstance(v, str) or isinstance(v, list), "values of  keys must be a string or list of strings."

                if isinstance(v, list):
                    new_value = []


                    for element in v:
                        assert isinstance(element, str), "values of keys must be a string"
                        converted_element = self._convert_value(element)

                        if converted_element not in new_value:
                            new_value.append(converted_element)
                else:
                    new_value = [self._convert_value(v)]


                new_key = self._convert_key(k)
                new_keys[new_key] = new_value

        self.keys = new_keys
