from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterSpan:
    """
    Half-open character range ``[start, end)`` inside a caption.

    Attributes
    ----------
    start : int
        Index of the first character.
    end : int
        Index one past the last character.

    Raises
    ------
    ValueError
        If the range is empty or negative.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"CharacterSpan requires 0 <= start < end. got [{self.start}, {self.end})")

    def contains(self, start: int, end: int) -> bool:
        """
        Check whether a non-empty range lies inside this span.

        Parameters
        ----------
        start : int
            Start of the range to test.
        end : int
            End of the range to test.

        Returns
        -------
        bool
            True if ``[start, end)`` is non-empty and inside this span.
        """
        return self.start <= start < end <= self.end
