from enum import StrEnum


class PromptKind(StrEnum):
    """
    Kind of prompt, i.e. which inputs tell the model what each class looks like.

    Attributes
    ----------
    TEXT : str
        Every class is queried by its name.
    TEXT_VISUAL : str
        Classes with visual references are queried by those image regions, the others by their name.
    """

    TEXT = "text"
    TEXT_VISUAL = "text_visual"
