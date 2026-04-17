from __future__ import annotations

from functools import update_wrapper
from typing import Any, Callable
from weakref import WeakKeyDictionary


class MultiDispatchDescriptor:
    """
    Inheritance-aware multi-dispatch descriptor.

    Supports:
    - instance methods
    - class methods
    - static methods

    Dispatch is based on the runtime types of all positional arguments
    that are explicitly passed by the caller.
    """

    def __init__(self, func: Callable[..., Any], *, binding: str) -> None:
        if binding not in {"instance", "class", "static"}:
            raise ValueError(f"Unsupported binding mode: {binding}")

        self._name = func.__name__
        self._default_impl = func
        self._binding = binding

        # One overload registry per owning class.
        self._registries: WeakKeyDictionary[type, dict[tuple[type, ...], Callable[..., Any]]] = (
            WeakKeyDictionary()
        )

        update_wrapper(self, func)

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name
        self._ensure_registry(owner)

    def register(self, *types: type) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Register an overload signature.

        Usage inside the same class:
            @parse.register(bytes, int)
            def _parse_bytes_int(self, data, n): ...

        Usage inside subclasses:
            @BaseParser.parse.register(bytes, int)
            def _parse_bytes_int(self, data, n): ...
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            entries = getattr(func, "__multidispatch_entries__", None)
            if entries is None:
                entries = []
                setattr(func, "__multidispatch_entries__", entries)
            entries.append((self, types))
            return func

        return decorator

    def _ensure_registry(self, owner: type) -> dict[tuple[type, ...], Callable[..., Any]]:
        if owner in self._registries:
            return self._registries[owner]

        registry: dict[tuple[type, ...], Callable[..., Any]] = {}

        # Inherit overloads from base classes first.
        for base in reversed(owner.__mro__[1:]):
            base_registry = self._registries.get(base)
            if base_registry:
                registry.update(base_registry)

        # Add/override overloads declared directly in this class body.
        for value in owner.__dict__.values():
            entries = getattr(value, "__multidispatch_entries__", ())
            for descriptor, types in entries:
                if descriptor is self:
                    registry[types] = value

        self._registries[owner] = registry
        return registry

    def _resolve(self, owner: type, args: tuple[Any, ...]) -> Callable[..., Any]:
        registry = self._ensure_registry(owner)
        arg_types = tuple(type(arg) for arg in args)

        # Exact match first.
        exact = registry.get(arg_types)
        if exact is not None:
            return exact

        # Then compatible matches.
        matches: list[tuple[int, Callable[..., Any]]] = []

        for registered_types, func in registry.items():
            if len(registered_types) != len(args):
                continue

            if all(isinstance(arg, expected) for arg, expected in zip(args, registered_types)):
                score = self._specificity_score(args, registered_types)
                matches.append((score, func))

        if not matches:
            return self._default_impl

        matches.sort(key=lambda item: item[0], reverse=True)

        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            raise TypeError(
                f"Ambiguous overload for {owner.__name__}.{self._name}{arg_types}"
            )

        return matches[0][1]

    @staticmethod
    def _specificity_score(args: tuple[Any, ...], registered_types: tuple[type, ...]) -> int:
        score = 0
        for arg, expected in zip(args, registered_types):
            mro = type(arg).mro()
            try:
                distance = mro.index(expected)
            except ValueError:
                distance = 10_000
            score -= distance
        return score

    def __get__(self, instance: Any, owner: type | None = None) -> Callable[..., Any]:
        if owner is None:
            owner = type(instance)

        if self._binding == "instance":
            if instance is None:
                return self

            def bound(*args: Any, **kwargs: Any) -> Any:
                if kwargs:
                    raise TypeError(
                        f"{owner.__name__}.{self._name} does not support keyword-based dispatch"
                    )
                func = self._resolve(owner, args)
                return func(instance, *args)

            update_wrapper(bound, self._default_impl)
            return bound

        if self._binding == "class":
            def bound(*args: Any, **kwargs: Any) -> Any:
                if kwargs:
                    raise TypeError(
                        f"{owner.__name__}.{self._name} does not support keyword-based dispatch"
                    )
                func = self._resolve(owner, args)
                return func(owner, *args)

            update_wrapper(bound, self._default_impl)
            return bound

        if self._binding == "static":
            def bound(*args: Any, **kwargs: Any) -> Any:
                if kwargs:
                    raise TypeError(
                        f"{owner.__name__}.{self._name} does not support keyword-based dispatch"
                    )
                func = self._resolve(owner, args)
                return func(*args)

            update_wrapper(bound, self._default_impl)
            return bound

        raise RuntimeError(f"Unknown binding mode: {self._binding}")


def multidispatchmethod(func: Callable[..., Any]) -> MultiDispatchDescriptor:
    return MultiDispatchDescriptor(func, binding="instance")


def multidispatchclassmethod(func: Callable[..., Any]) -> MultiDispatchDescriptor:
    return MultiDispatchDescriptor(func, binding="class")


def multidispatchstaticmethod(func: Callable[..., Any]) -> MultiDispatchDescriptor:
    return MultiDispatchDescriptor(func, binding="static")


class MultiDispatchSupport:
    """
    Mixin that finalizes overload registrations for subclasses.

    Any class that uses inherited @Base.method.register(...) overloads
    should inherit from this mixin.
    """
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        seen: set[MultiDispatchDescriptor] = set()

        # Ensure registries for all visible descriptors in the MRO.
        for base in cls.__mro__:
            for value in base.__dict__.values():
                if isinstance(value, MultiDispatchDescriptor) and value not in seen:
                    value._ensure_registry(cls)
                    seen.add(value)

        # Also ensure registries for descriptors referenced by tagged methods
        # declared directly in this subclass body.
        for value in cls.__dict__.values():
            entries = getattr(value, "__multidispatch_entries__", ())
            for descriptor, _types in entries:
                if descriptor not in seen:
                    descriptor._ensure_registry(cls)
                    seen.add(descriptor)