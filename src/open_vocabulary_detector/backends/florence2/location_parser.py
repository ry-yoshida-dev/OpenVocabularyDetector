import re
from typing import ClassVar

import numpy as np

from ...array_types import FloatArray


class LocationParser:
    """
    Reads the boxes Florence-2 writes as location tokens into normalized XYXY boxes.

    Florence-2 quantizes each coordinate into ``BIN_COUNT`` bins and writes it as ``<loc_N>``; a box is four
    consecutive tokens (``cat<loc_8><loc_114><loc_497><loc_990>``). Open-vocabulary detection may also answer with
    polygons (``<poly><loc_x><loc_y>...</poly>``), whose points are bounded by a box; both forms may be mixed and
    are read in generation order. Each bin is read at its center, like the ``transformers`` post-processing, but
    without rounding to whole pixels.
    """

    BIN_COUNT: ClassVar[int] = 1000
    REGION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"<poly>.*?</poly>|(?:<loc_\d+>){4}")
    LOCATION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"<loc_(\d+)>")

    @classmethod
    def parse(cls, generated_text: str) -> FloatArray:
        """
        Extract every box of a generated answer.

        Parameters
        ----------
        generated_text : str
            Decoded generation, special tokens included.

        Returns
        -------
        FloatArray
            XYXY boxes relative to the image size, shape (N, 4).
        """
        bins: FloatArray = np.array(
            [bounds for match in cls.REGION_PATTERN.finditer(generated_text) if (bounds := cls._region_bounds(match))],
            dtype=np.float64,
        )
        return ((bins.reshape(-1, 4) + 0.5) / cls.BIN_COUNT).clip(0.0, 1.0)

    @classmethod
    def _region_bounds(cls, match: re.Match[str]) -> list[int]:
        locations: list[int] = [int(value) for value in cls.LOCATION_PATTERN.findall(match.group(0))]
        point_count: int = len(locations) // 2
        if point_count == 0:
            return []
        x_bins: list[int] = locations[0 : 2 * point_count : 2]
        y_bins: list[int] = locations[1 : 2 * point_count : 2]
        return [min(x_bins), min(y_bins), max(x_bins), max(y_bins)]
