from dataclasses import dataclass

from ..core import Prompt
from ..kind import PromptKind


@dataclass(frozen=True)
class TextPrompt(Prompt):
    """
    Prompt whose classes are queried by their names, e.g. ``"person"`` or ``"a photo of a red traffic cone"``.
    """

    @property
    def kind(self) -> PromptKind:
        return PromptKind.TEXT
