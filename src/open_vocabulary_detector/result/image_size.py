from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ImageSize:
    """
    Pixel size of an image.

    Attributes
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Raises
    ------
    ValueError
        If width or height is not positive.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be positive. got ({self.width}, {self.height})")

    @classmethod
    def from_image(cls, image: Image.Image) -> "ImageSize":
        """
        Read the size of a PIL image.

        Parameters
        ----------
        image : Image.Image
            Source image.

        Returns
        -------
        ImageSize
            Width and height of the image.
        """
        width, height = image.size
        return cls(width=width, height=height)
