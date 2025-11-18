"""
This module provides utility classes and functions for audio processing in the sdialog library.

The module includes various utility classes and enums that support audio generation,
room acoustics simulation, and voice database management. These utilities provide
standardized data structures and helper functions for audio-related operations.

Key Components:

  - RGBAColor: Color enumeration for visual representations
  - Furniture: 3D furniture model for room simulation
  - BodyPosture: Body posture height constants
  - WallMaterial, FloorMaterial, CeilingMaterial: Material enums for acoustics
  - SourceType, SpeakerSide: Audio source and speaker positioning enums
  - SourceVolume: Audio volume level enumeration
  - AudioUtils: Utility functions for audio processing
  - RoomMaterials: Room material configuration model
  - Role: Speaker role enumeration

Example:

    .. code-block:: python

        from sdialog.audio.utils import RGBAColor, Furniture, AudioUtils

        # Create furniture for room simulation
        chair = Furniture(
            name="office_chair",
            x=1.0, y=2.0, z=0.0,
            width=0.5, height=1.2, depth=0.5,
            color=RGBAColor.BLACK
        )

        # Process audio text
        clean_text = AudioUtils.remove_audio_tags("<speak>Hello world</speak>")
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import re
import logging
from enum import Enum
from pydantic import BaseModel

# Create a logger for the audio module
logger = logging.getLogger("sdialog.audio")


class RGBAColor(Enum):
    """
    RGBA color enumeration for visual representations in room simulation.

    This enum provides predefined RGBA color values for furniture and objects
    in room acoustics simulation. Each color is represented as a tuple of
    (red, green, blue, alpha) values, where alpha controls transparency.

    Color values are in the range 0-255 for RGB components and 0-100 for alpha.
    The alpha value of 50 provides semi-transparency for visual overlays.

    :ivar RED: Red color (255, 0, 0, 50).
    :vartype RED: tuple[int, int, int, int]
    :ivar GREEN: Green color (0, 255, 0, 50).
    :vartype GREEN: tuple[int, int, int, int]
    :ivar BLUE: Blue color (0, 0, 255, 50).
    :vartype BLUE: tuple[int, int, int, int]
    :ivar YELLOW: Yellow color (255, 255, 0, 50).
    :vartype YELLOW: tuple[int, int, int, int]
    :ivar PURPLE: Purple color (128, 0, 128, 50).
    :vartype PURPLE: tuple[int, int, int, int]
    :ivar ORANGE: Orange color (255, 165, 0, 50).
    :vartype ORANGE: tuple[int, int, int, int]
    :ivar PINK: Pink color (255, 192, 203, 50).
    :vartype PINK: tuple[int, int, int, int]
    :ivar BROWN: Brown color (165, 42, 42, 50).
    :vartype BROWN: tuple[int, int, int, int]
    :ivar GRAY: Gray color (128, 128, 128, 50).
    :vartype GRAY: tuple[int, int, int, int]
    :ivar BLACK: Black color (0, 0, 0, 50).
    :vartype BLACK: tuple[int, int, int, int]
    :ivar WHITE: White color (255, 255, 255, 50).
    :vartype WHITE: tuple[int, int, int, int]
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
    3D furniture model for room acoustics simulation.

    This class represents a piece of furniture in a 3D room environment,
    providing spatial positioning, dimensions, and visual properties.
    Furniture objects are used in room acoustics simulation to model
    acoustic obstacles and reflections.

    Key Features:

      - 3D spatial positioning (x, y, z coordinates)
      - 3D dimensions (width, height, depth)
      - Visual color representation
      - Acoustic modeling support

    :ivar name: Name identifier for the furniture piece.
    :vartype name: str
    :ivar x: X-axis position in meters (horizontal).
    :vartype x: float
    :ivar y: Y-axis position in meters (depth).
    :vartype y: float
    :ivar z: Z-axis position in meters (height, default: 0.0).
    :vartype z: float
    :ivar width: Width of the furniture in meters (x-axis dimension).
    :vartype width: float
    :ivar height: Height of the furniture in meters (z-axis dimension).
    :vartype height: float
    :ivar depth: Depth of the furniture in meters (y-axis dimension).
    :vartype depth: float
    :ivar color: RGBA color for visual representation (default: RED).
    :vartype color: RGBAColor
    """

    name: str

    x: float  # x-axis in meters
    y: float  # y-axis in meters
    z: float = 0.0  # z-axis in meters

    width: float  # width in meters
    height: float  # height in meters
    depth: float  # depth in meters

    color: RGBAColor = RGBAColor.RED

    def get_top_z(self) -> float:
        """
        Calculates the top Z-coordinate of the furniture.

        This method returns the highest Z-coordinate of the furniture,
        which is useful for collision detection and spatial calculations
        in room acoustics simulation.

        :return: The top Z-coordinate (z + height).
        :rtype: float
        """
        return self.z + self.height


class BodyPosture(Enum):
    """
    Body posture height enumeration for speaker positioning in room simulation.

    This enum provides standard height values for different body postures,
    which are used to position speakers at realistic heights in room
    acoustics simulation. These values represent the approximate height
    of a person's mouth/head in different postures.

    :ivar SITTING: Sitting posture height in meters (0.5m).
    :vartype SITTING: float
    :ivar STANDING: Standing posture height in meters (1.7m).
    :vartype STANDING: float
    """
    SITTING = 0.5
    STANDING = 1.7


class WallMaterial(str, Enum):
    """
    Wall material enumeration for room acoustics simulation.

    This enum provides a comprehensive list of common wall materials with
    their corresponding acoustic properties. Each material has specific
    absorption coefficients that affect sound reflection and reverberation
    in room acoustics simulation.

    Materials range from highly reflective (hard surfaces) to highly
    absorptive (soft materials), providing realistic acoustic modeling
    for different room types and environments.

    :ivar HARD_SURFACE: Hard surface material with low absorption.
    :vartype HARD_SURFACE: str
    :ivar BRICKWORK: Brickwork material with moderate absorption.
    :vartype BRICKWORK: str
    :ivar ROUGH_CONCRETE: Rough concrete with moderate absorption.
    :vartype ROUGH_CONCRETE: str
    :ivar UNPAINTED_CONCRETE: Unpainted concrete with moderate absorption.
    :vartype UNPAINTED_CONCRETE: str
    :ivar ROUGH_LIME_WASH: Rough lime wash with moderate absorption.
    :vartype ROUGH_LIME_WASH: str
    :ivar SMOOTH_BRICKWORK_FLUSH_POINTING: Smooth brickwork with flush pointing.
    :vartype SMOOTH_BRICKWORK_FLUSH_POINTING: str
    :ivar SMOOTH_BRICKWORK_10MM_POINTING: Smooth brickwork with 10mm pointing.
    :vartype SMOOTH_BRICKWORK_10MM_POINTING: str
    :ivar BRICK_WALL_ROUGH: Rough brick wall material.
    :vartype BRICK_WALL_ROUGH: str
    :ivar CERAMIC_TILES: Ceramic tiles with moderate absorption.
    :vartype CERAMIC_TILES: str
    :ivar LIMESTONE_WALL: Limestone wall material.
    :vartype LIMESTONE_WALL: str
    :ivar REVERB_CHAMBER: Reverb chamber material with high reflection.
    :vartype REVERB_CHAMBER: str
    :ivar CONCRETE_FLOOR: Concrete floor material.
    :vartype CONCRETE_FLOOR: str
    :ivar MARBLE_FLOOR: Marble floor material with low absorption.
    :vartype MARBLE_FLOOR: str
    :ivar PLASTERBOARD: Plasterboard with moderate absorption.
    :vartype PLASTERBOARD: str
    :ivar WOODEN_LINING: Wooden lining with moderate absorption.
    :vartype WOODEN_LINING: str
    :ivar WOOD_1_6CM: 1.6cm wood material.
    :vartype WOOD_1_6CM: str
    :ivar PLYWOOD_THIN: Thin plywood material.
    :vartype PLYWOOD_THIN: str
    :ivar WOOD_16MM: 16mm wood material.
    :vartype WOOD_16MM: str
    :ivar AUDIENCE_FLOOR: Audience floor material.
    :vartype AUDIENCE_FLOOR: str
    :ivar STAGE_FLOOR: Stage floor material.
    :vartype STAGE_FLOOR: str
    :ivar WOODEN_DOOR: Wooden door material.
    :vartype WOODEN_DOOR: str
    """

    HARD_SURFACE = "hard_surface"
    BRICKWORK = "brickwork"
    ROUGH_CONCRETE = "rough_concrete"
    UNPAINTED_CONCRETE = "unpainted_concrete"
    ROUGH_LIME_WASH = "rough_lime_wash"
    SMOOTH_BRICKWORK_FLUSH_POINTING = "smooth_brickwork_flush_pointing"
    SMOOTH_BRICKWORK_10MM_POINTING = "smooth_brickwork_10mm_pointing"
    BRICK_WALL_ROUGH = "brick_wall_rough"
    CERAMIC_TILES = "ceramic_tiles"
    LIMESTONE_WALL = "limestone_wall"
    REVERB_CHAMBER = "reverb_chamber"
    CONCRETE_FLOOR = "concrete_floor"
    MARBLE_FLOOR = "marble_floor"
    PLASTERBOARD = "plasterboard"
    WOODEN_LINING = "wooden_lining"
    WOOD_1_6CM = "wood_1.6cm"
    PLYWOOD_THIN = "plywood_thin"
    WOOD_16MM = "wood_16mm"
    AUDIENCE_FLOOR = "audience_floor"
    STAGE_FLOOR = "stage_floor"
    WOODEN_DOOR = "wooden_door"


class FloorMaterial(str, Enum):
    """
    Floor materials affecting acoustics
    """

    LINOLEUM_ON_CONCRETE = "linoleum_on_concrete"
    CARPET_COTTON = "carpet_cotton"
    CARPET_TUFTED_9_5MM = "carpet_tufted_9.5mm"
    CARPET_THIN = "carpet_thin"
    CARPET_6MM_CLOSED_CELL_FOAM = "carpet_6mm_closed_cell_foam"
    CARPET_6MM_OPEN_CELL_FOAM = "carpet_6mm_open_cell_foam"
    CARPET_TUFTED_9M = "carpet_tufted_9m"
    FELT_5MM = "felt_5mm"
    CARPET_SOFT_10MM = "carpet_soft_10mm"
    CARPET_HAIRY = "carpet_hairy"
    CARPET_RUBBER_5MM = "carpet_rubber_5mm"
    CARPET_1_35_KG_M2 = "carpet_1.35_kg_m2"
    COCOS_FIBRE_ROLL_29MM = "cocos_fibre_roll_29mm"


class CeilingMaterial(str, Enum):
    """
    Floor materials affecting acoustics
    """

    PLASTERBOARD = "ceiling_plasterboard"
    FIBRE_ABSORBER = "ceiling_fibre_absorber"
    FISSURED_TILE = "ceiling_fissured_tile"
    PERFORATED_GYPSUM_BOARD = "ceiling_perforated_gypsum_board"
    MELAMINE_FOAM = "ceiling_melamine_foam"
    METAL_PANEL = "ceiling_metal_panel"


class SourceType(str, Enum):
    """
    Type of the audio source
    """
    BACKGROUND = "no_type"
    ROOM = "room-"
    EVENT = "soundevent-"


class SpeakerSide(str, Enum):
    """
    Side of the speaker relative to the furniture
    """
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"


class SourceVolume(Enum):
    """
    Volume of the audio source
    """

    VERY_LOW = 0.0000001
    LOW = 0.01
    MEDIUM = 0.02
    HIGH = 0.05
    VERY_HIGH = 0.07
    EXTREMELY_HIGH = 0.10


class AudioUtils:
    """
    Utility class for audio processing operations.

    This class provides static utility methods for common audio processing
    tasks, including text preprocessing for TTS engines and audio data
    manipulation. These utilities help ensure consistent audio processing
    across different components of the sdialog library.

    Key Features:

      - Text preprocessing for TTS engines
      - Audio tag removal and cleaning
      - Audio data validation and processing
      - Common audio operations and transformations
    """

    @staticmethod
    def remove_audio_tags(text: str) -> str:
        """
        Removes audio-specific tags and formatting from text.

        This method cleans text by removing various types of audio tags
        and formatting that might interfere with TTS generation. It removes
        XML-style tags, asterisks, and other formatting elements that are
        commonly used in audio markup languages.

        Supported tag formats:
            - XML-style tags: <tag>content</tag>
            - Asterisks: *text*
            - Other formatting elements

        :param text: The text to clean of audio tags and formatting.
        :type text: str
        :return: The cleaned text with audio tags and formatting removed.
        :rtype: str
        """
        return re.sub(r'<[^>]*>', '', text).replace("*", "")


class RoomMaterials(BaseModel):
    """
    Room materials configuration for acoustics simulation.

    This class defines the material properties for different surfaces
    in a room, which are used to model acoustic behavior in room
    acoustics simulation. Each surface (ceiling, walls, floor) can
    have different materials with specific absorption coefficients.

    Key Features:

      - Configurable ceiling, wall, and floor materials
      - Acoustic property modeling for each surface
      - Realistic room acoustics simulation support
      - Material-specific absorption coefficients

    :ivar ceiling: Ceiling material (default: FIBRE_ABSORBER).
    :vartype ceiling: CeilingMaterial
    :ivar walls: Wall material (default: WOODEN_LINING).
    :vartype walls: WallMaterial
    :ivar floor: Floor material (default: CARPET_HAIRY).
    :vartype floor: FloorMaterial
    """
    ceiling: CeilingMaterial = CeilingMaterial.FIBRE_ABSORBER
    walls: WallMaterial = WallMaterial.WOODEN_LINING
    floor: FloorMaterial = FloorMaterial.CARPET_HAIRY


class Role(str, Enum):
    """
    Speaker role enumeration for dialogue management.

    This enum defines the roles that speakers can have in a dialogue,
    providing a standardized way to identify and manage different
    participants in audio dialogue generation and processing.

    Key Features:

      - Standardized speaker role identification
      - Support for multi-speaker dialogues
      - Role-based voice assignment and management
      - Dialogue structure organization

    :ivar SPEAKER_1: First speaker in the dialogue.
    :vartype SPEAKER_1: str
    :ivar SPEAKER_2: Second speaker in the dialogue.
    :vartype SPEAKER_2: str
    """
    SPEAKER_1 = "speaker_1"
    SPEAKER_2 = "speaker_2"

    def __str__(self):
        return self.value


def default_dscaper_datasets() -> list[str]:
    """
    Default dSCAPER datasets
    """
    return ["sdialog/background", "sdialog/foreground"]
