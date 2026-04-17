from enum import Enum


class EpochRotation(Enum):
    PACKET = "packet"
    CALL = "call"
    STREAM = "stream"
    NEVER = "never"
