import torch
from torchvision.ops import box_convert, box_iou, generalized_box_iou


class OwlViTBoxQuerySelector:
    """
    Picks the patch embedding that represents a box of a reference image (image-guided OWL-ViT).

    Follows the Hugging Face ``embed_image_query`` heuristic for an arbitrary target box instead of the
    whole image: patches whose predicted box overlaps the target by at least ``RELATIVE_IOU_THRESHOLD``
    of the best IoU are candidates, and the candidate least similar to the mean patch embedding
    (the least background-like one) is chosen. When no patch box overlaps the target, the patch with
    the highest generalized IoU is chosen.
    """

    RELATIVE_IOU_THRESHOLD: float = 0.8

    @classmethod
    def select(
        cls,
        target_xyxy: torch.Tensor,
        predicted_cxcywh: torch.Tensor,
        class_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select one patch embedding per target box.

        Parameters
        ----------
        target_xyxy : torch.Tensor
            Target boxes normalized to ``[0, 1]``, shape (N, 4).
        predicted_cxcywh : torch.Tensor
            Box predicted for each patch, normalized center-size, shape (P, 4).
        class_embeddings : torch.Tensor
            Class embedding of each patch, shape (P, D).

        Returns
        -------
        torch.Tensor
            Selected embedding of each target box, shape (N, D).
        """
        predicted_xyxy: torch.Tensor = box_convert(predicted_cxcywh.float(), in_fmt="cxcywh", out_fmt="xyxy")
        target_boxes: torch.Tensor = target_xyxy.float().to(predicted_xyxy.device)
        embeddings: torch.Tensor = class_embeddings.float()
        mean_embedding: torch.Tensor = embeddings.mean(dim=0)
        overlaps: torch.Tensor = box_iou(target_boxes, predicted_xyxy)
        selected_embeddings: list[torch.Tensor] = []
        for target_index in range(target_boxes.shape[0]):
            target_overlaps: torch.Tensor = overlaps[target_index]
            if not bool((target_overlaps > 0).any()):
                generalized_overlaps: torch.Tensor = generalized_box_iou(
                    target_boxes[target_index : target_index + 1], predicted_xyxy
                )[0]
                selected_embeddings.append(embeddings[generalized_overlaps.argmax()])
                continue
            candidate_indices: torch.Tensor = (
                (target_overlaps >= target_overlaps.max() * cls.RELATIVE_IOU_THRESHOLD).nonzero().squeeze(1)
            )
            similarities: torch.Tensor = embeddings[candidate_indices] @ mean_embedding
            selected_embeddings.append(embeddings[candidate_indices[similarities.argmin()]])
        return torch.stack(selected_embeddings)
