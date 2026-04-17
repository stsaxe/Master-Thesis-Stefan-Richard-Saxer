from __future__ import annotations

import hashlib
import hmac
from ble.pseudomization.AbstractRandomizer import AbstractRandomizer
from ble.utils.HelperMethods import HelperMethods




class HexPseudomizer(HelperMethods, AbstractRandomizer):
    def pseudomize(self, token: str, length: int) -> str:
        if not isinstance(length, int) or length < 0:
            raise ValueError("length must be a non-negative int")

        HelperMethods.check_valid_string(token)

        if length == 0:
            return ""

        subkey = self._subkey(token)

        out = []
        counter = 0
        while sum(len(part) for part in out) < length:
            block = hmac.new(
                subkey,
                counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).hexdigest()
            out.append(block)

            counter += 1

        value = "".join(out)[:length].upper()

        return HelperMethods.clean_hex_value(value)

