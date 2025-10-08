import re
from enum import Enum
from pydantic import BaseModel


class RGBAColor(Enum):
    """
    RGBA color (red, green, blue, alpha).
    """
    RED = (255, 0, 0, 50)
    GREEN = (0, 255, 0, 50)
    BLUE = (0, 0, 255, 50)
    YELLOW = (255, 255, 0, 50)
    PURPLE = (128, 0, 128, 50)
    ORANGE = (255, 165, 0, 50)
    PINK = (255, 192, 203, 50)
    BROWN = (165, 42, 42, 50)
    GRAY = (128, 128, 128, 50)
    BLACK = (0, 0, 0, 50)
    WHITE = (255, 255, 255, 50)


class Furniture(BaseModel):
    """
    Furniture in the room.
    """

    name: str

    x: float  # x-axis in meters
    y: float  # y-axis in meters

    width: float  # width in meters
    height: float  # height in meters
    depth: float  # depth in meters

    color: RGBAColor = RGBAColor.RED


# TODO: Add Float as heritance for the serialization of the model
class BodyPosture(Enum):
    """
    Body posture height in meters.
    """
    SITTING = 0.5
    STANDING = 1.7


class AudioUtils:
    """
    Audio utilities.
    """

    @staticmethod
    def remove_audio_tags(text: str) -> str:
        """
        Remove all the tags that use those formatting: <>, {}, (), []
        """
        return re.sub(r'<[^>]*>', '', text).replace("*", "")
