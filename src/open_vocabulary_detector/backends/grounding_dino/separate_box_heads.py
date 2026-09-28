from typing import ClassVar, cast

from transformers import MMGroundingDinoForObjectDetection


class MMGroundingDinoWithSeparateBoxHeads(MMGroundingDinoForObjectDetection):
    """
    MM-Grounding-DINO that keeps one box head per decoder layer when its checkpoint says so.

    ``transformers`` ties the box heads of every decoder layer to the first one, so the separately trained heads of
    the published MM-Grounding-DINO and LLMDet checkpoints (``decoder_bbox_embed_share: false`` in their
    ``config.json``) are overwritten after loading and predicted boxes drift. Like ``GroundingDinoForObjectDetection``,
    this model drops that tie when the checkpoint does not share its box heads; a checkpoint without the flag keeps
    the ``transformers`` behavior.
    """

    BOX_HEAD_SHARE_ATTRIBUTE: ClassVar[str] = "decoder_bbox_embed_share"
    SHARED_BOX_HEAD_PATTERN: ClassVar[str] = r"bbox_embed.(?![0])\d+"

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict[str, str]:
        """
        Tied parameter names, without the box head tie when the checkpoint keeps separate box heads.

        ``transformers`` registers the result while initializing the model, before the weights are loaded.

        Parameters
        ----------
        all_submodels : bool, optional
            Whether to include the ties of every submodel.

        Raises
        ------
        RuntimeError
            If the checkpoint keeps separate box heads but ``transformers`` no longer declares the box head tie as
            ``SHARED_BOX_HEAD_PATTERN``, so the tie could not be dropped.

        Returns
        -------
        dict[str, str]
            Source parameter name of every tied target parameter name.
        """
        is_box_head_shared: object = getattr(self.config, self.BOX_HEAD_SHARE_ATTRIBUTE, True)
        if is_box_head_shared is False:
            if self.SHARED_BOX_HEAD_PATTERN not in MMGroundingDinoForObjectDetection._tied_weights_keys:
                raise RuntimeError(
                    f"transformers does not tie the box heads as {self.SHARED_BOX_HEAD_PATTERN!r} anymore; "
                    + "separate box heads of the checkpoint cannot be kept"
                )
            self._tied_weights_keys = {
                target: source
                for target, source in MMGroundingDinoForObjectDetection._tied_weights_keys.items()
                if target != self.SHARED_BOX_HEAD_PATTERN
            }
        return cast(dict[str, str], super().get_expanded_tied_weights_keys(all_submodels=all_submodels))
