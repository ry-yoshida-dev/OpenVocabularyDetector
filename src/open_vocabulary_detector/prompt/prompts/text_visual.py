from dataclasses import dataclass

from ..core import Prompt
from ..kind import PromptKind
from ..visual_reference import VisualReference


@dataclass(frozen=True)
class TextVisualPrompt(Prompt):
    """
    Prompt mixing visual and text queries.

    Classes shown in at least one visual reference are queried by those image regions
    (boxes of the same class are averaged); the remaining classes are queried by their names.

    Attributes
    ----------
    visual_references : tuple[VisualReference, ...]
        Reference images with boxes labelled by class id.

    Raises
    ------
    ValueError
        If there is no visual reference or a reference points to a class id outside ``class_names``.
    """

    visual_references: tuple[VisualReference, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.visual_references:
            raise ValueError("visual_references must contain at least one reference; use TextPrompt otherwise.")
        class_count: int = len(self.class_names)
        for reference in self.visual_references:
            if bool(((reference.class_ids < 0) | (reference.class_ids >= class_count)).any()):
                raise ValueError(f"class_ids must be in [0, {class_count}). got {reference.class_ids.tolist()}")

    @property
    def kind(self) -> PromptKind:
        return PromptKind.TEXT_VISUAL

    @property
    def visual_class_ids(self) -> frozenset[int]:
        """
        Classes queried by visual references.

        Returns
        -------
        frozenset[int]
            Class ids appearing in at least one reference.
        """
        return frozenset(
            int(class_id) for reference in self.visual_references for class_id in reference.class_ids.tolist()
        )

    @property
    def text_class_ids(self) -> tuple[int, ...]:
        """
        Classes queried by their names.

        Returns
        -------
        tuple[int, ...]
            Class ids without any visual reference, in class-id order.
        """
        visual_class_ids: frozenset[int] = self.visual_class_ids
        return tuple(class_id for class_id in range(len(self.class_names)) if class_id not in visual_class_ids)
