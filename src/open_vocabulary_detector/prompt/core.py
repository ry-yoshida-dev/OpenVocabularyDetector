from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field

import numpy as np

from ..array_types import IntArray
from .kind import PromptKind
from .queries import PromptQuery, TextQuery, VisualQuery
from .visual_reference import VisualReference


@dataclass(frozen=True)
class Prompt:
    """
    Output classes to detect, each with the queries the model scores for it.

    Keys are the output class names (class id = key order); values are the queries of that class, text phrases,
    visual references or both::

        Prompt({
            "car": (TextQuery("car"), TextQuery("suv"), VisualQuery((van_reference_1, van_reference_2))),
            "person": (TextQuery("person"),),
        })

    The model scores every query, each box keeps its best one, and the detection is reported under the class of
    that query, together with the query itself.

    Attributes
    ----------
    class_names : tuple[str, ...]
        Output class names in class-id order; surrounding whitespace is stripped.
    queries : tuple[PromptQuery, ...]
        Queries in query-id order, grouped by class.
    query_class_ids : tuple[int, ...]
        Class id of each query, in query-id order.

    Raises
    ------
    ValueError
        If there is no class, a class name is blank or duplicated, a class has no query, a phrase or visual query
        is given twice (phrases compared ignoring case), or one visual reference is used by two classes.
    """

    class_queries: InitVar[Mapping[str, Sequence[PromptQuery]]]
    class_names: tuple[str, ...] = field(init=False)
    queries: tuple[PromptQuery, ...] = field(init=False)
    query_class_ids: tuple[int, ...] = field(init=False)

    def __post_init__(self, class_queries: Mapping[str, Sequence[PromptQuery]]) -> None:
        if not class_queries:
            raise ValueError("a prompt needs at least one class.")
        class_names: list[str] = []
        queries: list[PromptQuery] = []
        query_class_ids: list[int] = []
        for class_name, queries_of_class in class_queries.items():
            stripped_name: str = class_name.strip()
            if not stripped_name:
                raise ValueError(f"class names must not be blank. got {list(class_queries)}")
            if stripped_name in class_names:
                raise ValueError(f"class names must be unique. got {list(class_queries)}")
            if not queries_of_class:
                raise ValueError(f"{stripped_name!r} needs at least one query.")
            query_class_ids.extend([len(class_names)] * len(queries_of_class))
            class_names.append(stripped_name)
            queries.extend(queries_of_class)
        self._validate_distinct_queries(queries, query_class_ids)
        object.__setattr__(self, "class_names", tuple(class_names))
        object.__setattr__(self, "queries", tuple(queries))
        object.__setattr__(self, "query_class_ids", tuple(query_class_ids))

    @classmethod
    def from_texts(cls, class_texts: Mapping[str, Sequence[str]]) -> "Prompt":
        """
        Build a prompt querying every class by text phrases only.

        Parameters
        ----------
        class_texts : Mapping[str, Sequence[str]]
            Phrases of each class keyed by class name, e.g. ``{"car": ("car", "suv"), "person": ("person",)}``.

        Returns
        -------
        Prompt
            One ``TextQuery`` per phrase.
        """
        return cls({class_name: tuple(TextQuery(text) for text in texts) for class_name, texts in class_texts.items()})

    @classmethod
    def from_class_names(cls, class_names: Sequence[str]) -> "Prompt":
        """
        Build a prompt querying every class by its own name.

        Parameters
        ----------
        class_names : Sequence[str]
            Class names in class-id order, e.g. ``("person", "dog")``.

        Returns
        -------
        Prompt
            One ``TextQuery`` per class, reading the class name.
        """
        return cls.from_texts({class_name: (class_name,) for class_name in class_names})

    @property
    def kinds(self) -> frozenset[PromptKind]:
        """
        Kinds of the queries in the prompt.

        Returns
        -------
        frozenset[PromptKind]
            ``TEXT`` if any text query is given, ``VISUAL`` if any visual query is given.
        """
        return frozenset(self._kind_of(query) for query in self.queries)

    @property
    def query_texts(self) -> tuple[str, ...]:
        """
        Phrase of every query, for models that read text only.

        Raises
        ------
        ValueError
            If the prompt has a visual query.

        Returns
        -------
        tuple[str, ...]
            Phrases in query-id order.
        """
        texts: list[str] = []
        for query in self.queries:
            match query:
                case TextQuery(text=text):
                    texts.append(text)
                case VisualQuery():
                    raise ValueError("the prompt has visual queries; only text queries can be read as phrases.")
        return tuple(texts)

    @property
    def query_names(self) -> tuple[str, ...]:
        """
        Display name of every query, for models that name each query they score.

        Returns
        -------
        tuple[str, ...]
            Phrase of a text query, class name of a visual query, in query-id order.
        """
        return tuple(
            self._query_name(query, class_id)
            for query, class_id in zip(self.queries, self.query_class_ids, strict=True)
        )

    def class_ids_of(self, query_ids: IntArray) -> IntArray:
        """
        Class of each query id.

        Parameters
        ----------
        query_ids : IntArray
            Indices into ``queries``, any shape.

        Returns
        -------
        IntArray
            Class ids, same shape as ``query_ids``.
        """
        query_class_ids: IntArray = np.array(self.query_class_ids, dtype=np.int64)
        return query_class_ids[np.asarray(query_ids, dtype=np.int64)]

    def _query_name(self, query: PromptQuery, class_id: int) -> str:
        match query:
            case TextQuery(text=text):
                return text
            case VisualQuery():
                return self.class_names[class_id]

    @staticmethod
    def _kind_of(query: PromptQuery) -> PromptKind:
        match query:
            case TextQuery():
                return PromptKind.TEXT
            case VisualQuery():
                return PromptKind.VISUAL

    @staticmethod
    def _validate_distinct_queries(queries: Sequence[PromptQuery], query_class_ids: Sequence[int]) -> None:
        folded_texts: set[str] = set()
        visual_queries: set[VisualQuery] = set()
        reference_class_ids: dict[VisualReference, int] = {}
        for query, class_id in zip(queries, query_class_ids, strict=True):
            match query:
                case TextQuery(text=text):
                    if text.casefold() in folded_texts:
                        raise ValueError(f"text query {text!r} is given more than once.")
                    folded_texts.add(text.casefold())
                case VisualQuery(references=references):
                    if query in visual_queries:
                        raise ValueError("a visual query is given more than once.")
                    visual_queries.add(query)
                    for reference in references:
                        if reference_class_ids.setdefault(reference, class_id) != class_id:
                            raise ValueError("a visual reference must belong to only one class.")
