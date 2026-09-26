from collections.abc import Sequence
from dataclasses import dataclass

from .character_span import CharacterSpan


@dataclass(frozen=True)
class GroundingCaption:
    """
    Grounding DINO caption built from class names, with the character span of each class.

    Attributes
    ----------
    text : str
        Caption fed to the tokenizer, e.g. ``"cat . traffic cone ."``.
    class_spans : tuple[CharacterSpan, ...]
        Span of each class inside ``text``, in class-id order.
    """

    text: str
    class_spans: tuple[CharacterSpan, ...]

    @classmethod
    def from_class_names(cls, class_names: Sequence[str]) -> "GroundingCaption":
        """
        Join class names into a caption.

        Class names are lower-cased, separated by ``" . "`` and terminated by ``" ."``.

        Parameters
        ----------
        class_names : Sequence[str]
            Class names in class-id order.

        Returns
        -------
        GroundingCaption
            Caption and per-class spans.
        """
        text: str = ""
        class_spans: list[CharacterSpan] = []
        for class_name in class_names:
            start: int = len(text)
            text += class_name.lower()
            class_spans.append(CharacterSpan(start=start, end=len(text)))
            text += " . "
        return cls(text=text.rstrip(), class_spans=tuple(class_spans))
