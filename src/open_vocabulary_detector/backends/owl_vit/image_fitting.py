from collections.abc import Sequence
from enum import StrEnum

import torch

from ...result import ImageSize


class ImageFitting(StrEnum):
    """
    How an OWL processor fits an image into the square model input, which decides what predicted boxes are relative to.

    Attributes
    ----------
    STRETCH : str
        The image is resized to the input without padding (OWL-ViT), so input and image coordinates coincide.
    PAD_TO_SQUARE : str
        The image is padded at the bottom and right to a square before resizing (OWLv2), so the image covers only
        the top-left part of the input.
    """

    STRETCH = "stretch"
    PAD_TO_SQUARE = "pad_to_square"

    def coverage(self, image_size: ImageSize) -> torch.Tensor:
        """
        Part of the model input covered by an image, as scale factors of normalized XYXY or CXCYWH boxes.

        Parameters
        ----------
        image_size : ImageSize
            Size of the source image.

        Returns
        -------
        torch.Tensor
            ``(x, y, x, y)`` fractions of the input width and height taken by the image, shape (4,).
        """
        match self:
            case ImageFitting.STRETCH:
                return torch.ones(4, dtype=torch.float32)
            case ImageFitting.PAD_TO_SQUARE:
                side: int = max(image_size.width, image_size.height)
                width_fraction: float = image_size.width / side
                height_fraction: float = image_size.height / side
                return torch.tensor(
                    [width_fraction, height_fraction, width_fraction, height_fraction], dtype=torch.float32
                )

    def boxes_to_image(self, input_cxcywh: torch.Tensor, image_sizes: Sequence[ImageSize]) -> torch.Tensor:
        """
        Rescale predicted boxes from the model input to their images.

        Parameters
        ----------
        input_cxcywh : torch.Tensor
            CXCYWH boxes relative to the model input, shape (B, N, 4).
        image_sizes : Sequence[ImageSize]
            Size of each source image, one per batch entry.

        Raises
        ------
        ValueError
            If the number of image sizes differs from the batch size.

        Returns
        -------
        torch.Tensor
            Float32 CXCYWH boxes relative to their images, shape (B, N, 4); boxes in the padding exceed 1.
        """
        if len(image_sizes) != input_cxcywh.shape[0]:
            raise ValueError(f"expected {input_cxcywh.shape[0]} image sizes. got {len(image_sizes)}")
        coverages: torch.Tensor = torch.stack([self.coverage(image_size) for image_size in image_sizes])
        return input_cxcywh.float() / coverages.to(input_cxcywh.device)[:, None]

    def boxes_to_input(self, normalized_xyxy: torch.Tensor, image_size: ImageSize) -> torch.Tensor:
        """
        Map boxes of an image into the model input.

        Parameters
        ----------
        normalized_xyxy : torch.Tensor
            XYXY boxes relative to the image, shape (M, 4).
        image_size : ImageSize
            Size of the image.

        Returns
        -------
        torch.Tensor
            XYXY boxes relative to the model input, shape (M, 4), in the dtype of ``normalized_xyxy``.
        """
        return normalized_xyxy * self.coverage(image_size).to(normalized_xyxy)
