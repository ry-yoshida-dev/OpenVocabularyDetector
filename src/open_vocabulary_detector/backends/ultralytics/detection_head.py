from typing import Protocol


class DetectionHead(Protocol):
    """
    Last layer of an Ultralytics detection network, typed down to its class count.
    """

    nc: int
