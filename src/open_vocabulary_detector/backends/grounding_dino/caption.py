from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from .character_span import CharacterSpan


@dataclass(frozen=True)
class GroundingCaption:
    """
    Grounding DINO caption built from text queries, with the character span of each query.

    Grounding DINO treats ``"."`` and ``"?"`` as phrase delimiters, so they must not appear inside a query.

    Attributes
    ----------
    text : str
        Caption fed to the tokenizer, e.g. ``"cat . traffic cone ."``.
    query_spans : tuple[CharacterSpan, ...]
        Span of each query inside ``text``, in query-id order.
    """

    PHRASE_DELIMITERS: ClassVar[frozenset[str]] = frozenset({".", "?"})

    text: str
    query_spans: tuple[CharacterSpan, ...]

    @classmethod
    def from_queries(cls, text_queries: Sequence[str]) -> "GroundingCaption":
        """
        Join text queries into a caption.

        Queries are lower-cased, separated by ``" . "`` and terminated by ``" ."``.

        Parameters
        ----------
        text_queries : Sequence[str]
            Text queries in query-id order.

        Raises
        ------
        ValueError
            If a query contains a phrase delimiter, which would split it into several phrases.

        Returns
        -------
        GroundingCaption
            Caption and per-query spans.
        """
        text: str = ""
        query_spans: list[CharacterSpan] = []
        for text_query in text_queries:
            found_delimiters: frozenset[str] = cls.PHRASE_DELIMITERS.intersection(text_query)
            if found_delimiters:
                raise ValueError(
                    f"Grounding DINO text queries must not contain {sorted(found_delimiters)}, "
                    + f"which split a query into several phrases. got {text_query!r}"
                )
            start: int = len(text)
            text += text_query.lower()
            query_spans.append(CharacterSpan(start=start, end=len(text)))
            text += " . "
        return cls(text=text.rstrip(), query_spans=tuple(query_spans))
