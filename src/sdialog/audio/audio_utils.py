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
    z: float = 0.0  # z-axis in meters

    width: float  # width in meters
    height: float  # height in meters
    depth: float  # depth in meters

    color: RGBAColor = RGBAColor.RED

    def get_top_z(self) -> float:
        """
        Get the z-axis for the top of the furniture.
        """
        return self.z + self.height


# TODO: Add Float as heritance for the serialization of the model
class BodyPosture(Enum):
    """
    Body posture height in meters.
    """
    SITTING = 0.5
    STANDING = 1.7


class WallMaterial(str, Enum):
    """
    Common wall materials with typical absorption coefficients
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


class SourceVolume(Enum):
    """
    Volume of the audio source
    """

    VERY_LOW = 0.0000001
    LOW = 0.01
    MEDIUM = 0.02
    HIGH = 0.05
    VERY_HIGH = 0.07


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


class RoomMaterials(BaseModel):
    """
    Materials of the room.
    """
    ceiling: CeilingMaterial = CeilingMaterial.FIBRE_ABSORBER
    walls: WallMaterial = WallMaterial.WOODEN_LINING
    floor: FloorMaterial = FloorMaterial.CARPET_HAIRY
