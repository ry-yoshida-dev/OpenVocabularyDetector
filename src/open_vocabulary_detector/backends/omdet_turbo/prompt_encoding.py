from dataclasses import dataclass

import torch

from ...prompt import Prompt


@dataclass(frozen=True, eq=False)
class PromptEncoding:
    """
    Tokenized OmDet-Turbo prompt reused for every image batch of one ``Prompt``.

    OmDet-Turbo reads every text query as a class and the whole prompt as a task sentence
    (``"Detect car, suv, person."``); both are tokenized once per prompt.

    Attributes
    ----------
    prompt : Prompt
        Prompt the encoding was built from.
    class_input_ids : torch.Tensor
        Token ids of every text query, shape (Q, T).
    class_attention_mask : torch.Tensor
        Attention mask of every text query, shape (Q, T).
    task_input_ids : torch.Tensor
        Token ids of the task sentence, shape (1, T).
    task_attention_mask : torch.Tensor
        Attention mask of the task sentence, shape (1, T).

    Raises
    ------
    ValueError
        If the class rows do not match the queries of the prompt or the task is not a single row.
    """

    prompt: Prompt
    class_input_ids: torch.Tensor
    class_attention_mask: torch.Tensor
    task_input_ids: torch.Tensor
    task_attention_mask: torch.Tensor

    def __post_init__(self) -> None:
        query_count: int = len(self.prompt.queries)
        if self.class_input_ids.shape[0] != query_count or self.class_attention_mask.shape[0] != query_count:
            raise ValueError(
                f"class tokens must have {query_count} rows. got {tuple(self.class_input_ids.shape)} "
                + f"and {tuple(self.class_attention_mask.shape)}"
            )
        if self.task_input_ids.shape[0] != 1 or self.task_attention_mask.shape[0] != 1:
            raise ValueError(
                f"task tokens must have one row. got {tuple(self.task_input_ids.shape)} "
                + f"and {tuple(self.task_attention_mask.shape)}"
            )

    @property
    def query_count(self) -> int:
        """
        Number of text queries, i.e. classes as OmDet-Turbo sees them.

        Returns
        -------
        int
            Number of rows of ``class_input_ids``.
        """
        return int(self.class_input_ids.shape[0])

    def is_for(self, prompt: Prompt) -> bool:
        """
        Check whether the encoding was built from a prompt.

        Parameters
        ----------
        prompt : Prompt
            Prompt to compare with.

        Returns
        -------
        bool
            True if ``prompt`` equals the prompt of the encoding.
        """
        return self.prompt == prompt

    def to(self, device: torch.device) -> "PromptEncoding":
        """
        Move every tensor to a device.

        Parameters
        ----------
        device : torch.device
            Target device.

        Returns
        -------
        PromptEncoding
            Encoding whose tensors live on ``device``.
        """
        return PromptEncoding(
            prompt=self.prompt,
            class_input_ids=self.class_input_ids.to(device),
            class_attention_mask=self.class_attention_mask.to(device),
            task_input_ids=self.task_input_ids.to(device),
            task_attention_mask=self.task_attention_mask.to(device),
        )
