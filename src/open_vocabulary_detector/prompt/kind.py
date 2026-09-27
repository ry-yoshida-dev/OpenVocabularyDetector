from enum import StrEnum


class PromptKind(StrEnum):
    """
    Kind of prompt query, i.e. which input tells the model what a class looks like.

    A prompt may mix both kinds, even within one class; a backend accepts it when it supports every kind used.

    Attributes
    ----------
    TEXT : str
        Query by a text phrase.
    VISUAL : str
        Query by boxes in reference images.
    """

    TEXT = "text"
    VISUAL = "visual"
