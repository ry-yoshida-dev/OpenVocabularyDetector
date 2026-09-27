from dataclasses import dataclass


@dataclass(frozen=True)
class TextQuery:
    """
    Text phrase querying a class, e.g. ``TextQuery("suv")`` for ``"car"``.

    Attributes
    ----------
    text : str
        Phrase the model reads; surrounding whitespace is stripped.

    Raises
    ------
    ValueError
        If ``text`` is blank.
    """

    text: str

    def __post_init__(self) -> None:
        stripped_text: str = self.text.strip()
        if not stripped_text:
            raise ValueError("text must not be blank.")
        object.__setattr__(self, "text", stripped_text)
