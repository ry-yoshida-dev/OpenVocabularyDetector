from abc import ABC, abstractmethod
from dataclasses import dataclass

from .kind import PromptKind


@dataclass(frozen=True)
class Prompt(ABC):
    """
    Base of every prompt: the classes to detect.

    Class ids of detections index ``class_names``. Subclasses add only the inputs their kind needs.

    Attributes
    ----------
    class_names : tuple[str, ...]
        Class names in class-id order; surrounding whitespace is stripped.

    Raises
    ------
    ValueError
        If ``class_names`` is empty, contains a blank name, or contains duplicates.
    """

    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.class_names:
            raise ValueError("class_names must contain at least one name.")
        stripped_names: tuple[str, ...] = tuple(name.strip() for name in self.class_names)
        if any(not name for name in stripped_names):
            raise ValueError(f"class_names must not contain blank names. got {self.class_names}")
        if len(set(stripped_names)) != len(stripped_names):
            raise ValueError(f"class_names must be unique. got {self.class_names}")
        object.__setattr__(self, "class_names", stripped_names)

    @property
    @abstractmethod
    def kind(self) -> PromptKind:
        """
        Kind of the prompt.

        Returns
        -------
        PromptKind
            Inputs this prompt provides.
        """
