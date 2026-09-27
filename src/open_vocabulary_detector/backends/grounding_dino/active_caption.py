from dataclasses import dataclass

from ...prompt import Prompt
from .tokenized_caption import TokenizedCaption


@dataclass(frozen=True, eq=False)
class ActiveCaption:
    """
    Tokenized caption of the prompt a Grounding DINO detector is currently running.

    Attributes
    ----------
    prompt : Prompt
        Prompt the caption was built from.
    tokenized_caption : TokenizedCaption
        Caption tokens and query-to-token mask on the runtime device.
    """

    prompt: Prompt
    tokenized_caption: TokenizedCaption

    def is_for(self, prompt: Prompt) -> bool:
        """
        Check whether the caption was built from a prompt.

        Parameters
        ----------
        prompt : Prompt
            Prompt to compare with.

        Returns
        -------
        bool
            True if ``prompt`` equals the prompt of the caption.
        """
        return self.prompt == prompt
