from typing import Protocol

import torch


class TokenGenerator(Protocol):
    """
    Generation entry point of a ``transformers`` model returning token ids only.

    ``GenerationMixin.generate`` declares a ``self`` protocol that concrete models such as
    ``Florence2ForConditionalGeneration`` do not satisfy statically, although they do at runtime; models are viewed
    through this protocol to call it. Without ``return_dict_in_generate`` the call returns the token ids.
    """

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """
        Generate token ids for a batch of prompts and images.

        Parameters
        ----------
        input_ids : torch.Tensor
            Prompt token ids, shape (B, T).
        attention_mask : torch.Tensor
            Prompt attention mask, shape (B, T).
        pixel_values : torch.Tensor
            Preprocessed images, shape (B, C, H, W).
        max_new_tokens : int
            Maximum number of generated tokens.
        num_beams : int
            Beam search width.
        do_sample : bool
            Whether to sample instead of searching.

        Returns
        -------
        torch.Tensor
            Generated token ids, shape (B, L).
        """
        ...
