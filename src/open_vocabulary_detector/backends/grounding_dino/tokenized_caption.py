from dataclasses import dataclass

import torch
from transformers import BatchEncoding

from .caption import GroundingCaption


@dataclass(frozen=True, eq=False)
class TokenizedCaption:
    """
    Tokenized Grounding DINO caption reused for every image batch of a prompt.

    Grounding DINO fuses text and image features early, so the text encoder
    still runs per batch; tokenization and the query-to-token mapping are
    computed only once per prompt (see ``ActiveCaption``).

    Attributes
    ----------
    input_ids : torch.Tensor
        Token ids, shape (1, T).
    attention_mask : torch.Tensor
        Attention mask, shape (1, T).
    token_type_ids : torch.Tensor
        Token type ids, shape (1, T).
    query_token_mask : torch.Tensor
        Boolean mask, shape (Q, T); ``True`` where token t belongs to query q.

    Raises
    ------
    ValueError
        If the mask does not match the token count or a query maps to no token.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    query_token_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.query_token_mask.ndim != 2 or self.query_token_mask.shape[1] != self.token_count:
            raise ValueError(
                f"query_token_mask must have shape (Q, {self.token_count}). got {tuple(self.query_token_mask.shape)}"
            )
        if not bool(self.query_token_mask.any(dim=1).all()):
            raise ValueError("every query must map to at least one token.")

    @classmethod
    def from_encoding(cls, caption: GroundingCaption, encoding: BatchEncoding) -> "TokenizedCaption":
        """
        Map the character span of each query onto the tokens of an encoded caption.

        Parameters
        ----------
        caption : GroundingCaption
            Caption that was tokenized.
        encoding : BatchEncoding
            Tokenizer output for ``caption.text`` with ``return_offsets_mapping=True`` and PyTorch tensors.

        Returns
        -------
        TokenizedCaption
            Token tensors and query-to-token mask.
        """
        offsets: list[list[int]] = encoding["offset_mapping"][0].tolist()
        query_token_mask: torch.Tensor = torch.tensor(
            [[span.contains(start, end) for start, end in offsets] for span in caption.query_spans],
            dtype=torch.bool,
        )
        return cls(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding["token_type_ids"],
            query_token_mask=query_token_mask,
        )

    @property
    def token_count(self) -> int:
        """
        Number of tokens in the caption.

        Returns
        -------
        int
            Length T of the token sequence.
        """
        return int(self.input_ids.shape[1])

    def to(self, device: torch.device) -> "TokenizedCaption":
        """
        Move every tensor to a device.

        Parameters
        ----------
        device : torch.device
            Target device.

        Returns
        -------
        TokenizedCaption
            Caption whose tensors live on ``device``.
        """
        return TokenizedCaption(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_type_ids=self.token_type_ids.to(device),
            query_token_mask=self.query_token_mask.to(device),
        )

    def query_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Confidence of each query as the highest probability among its tokens.

        Parameters
        ----------
        logits : torch.Tensor
            Token logits of the predicted boxes, shape (B, N, L) with ``L >= T``.

        Returns
        -------
        torch.Tensor
            Query probabilities, shape (B, N, Q).
        """
        token_probabilities: torch.Tensor = logits[..., : self.token_count].float().sigmoid()
        return torch.stack(
            [token_probabilities[..., query_tokens].amax(dim=-1) for query_tokens in self.query_token_mask], dim=-1
        )
