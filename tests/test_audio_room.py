import pytest
import numpy as np
from sdialog.audio.room import (
    Position3D,
    Dimensions3D,
    Room,
    RoomRole,
    MaterialProperties,
    WallMaterial,
    RecordingDeviceSpec,
    RecordingDevice,
    EnvironmentalConditions
)
from sdialog.audio.room_generator import RoomGenerator, ROOM_SIZES


# Tests for room.py

def test_position3d():
    """Test Position3D dataclass."""
    pos = Position3D(1, 2, 3.5)
    assert pos.x == 1
    assert pos.y == 2
    assert pos.z == 3.5
    assert str(pos) == "pos: [1, 2, 3.5]"
    assert np.array_equal(pos.to_array(), np.array([1, 2, 3.5]))

    with pytest.raises(ValueError):
        Position3D(-1, 2, 3)


def test_dimensions3d():
    """Test Dimensions3D dataclass."""
    dims = Dimensions3D(width=4, length=5, height=3)
    assert dims.width == 4
    assert dims.volume == 60
    assert dims.floor_area == 20
    assert str(dims) == "dim: [4, 5, 3]"

    with pytest.raises(ValueError):
        Dimensions3D(0, 5, 3)

    dims_from_vol = Dimensions3D.from_volume(60)
    assert dims_from_vol.volume == pytest.approx(60)
    assert dims_from_vol.width == pytest.approx(7.66, abs=1e-2)
    assert dims_from_vol.length == pytest.approx(5.11, abs=1e-2)
    assert dims_from_vol.height == pytest.approx(1.53, abs=1e-2)


def test_room_creation():
    """Test Room class creation."""
    dims = Dimensions3D(4, 5, 3)
    room = Room(role=RoomRole.OFFICE, dimensions=dims, name="Test Office", reverberation_time_ratio=0.5)
    assert room.role == RoomRole.OFFICE
    assert room.dimensions == dims
    assert "Test Office" in room.name
    assert room.reverberation_time_ratio == 0.5
    assert "dimentions: [4, 5, 3]" in str(room)


def test_material_properties():
    """Test MaterialProperties dataclass."""
    mat = MaterialProperties(material_type=WallMaterial.DRYWALL)
    assert mat.absorption_coefficients[125] == 0.05


def test_recording_device_spec():
    """Test RecordingDeviceSpec dataclass."""
    spec = RecordingDeviceSpec(device_type=RecordingDevice.SMARTPHONE)
    assert spec.sensitivity == -38.0
    assert spec.snr == 50.0


def test_environmental_conditions():
    """Test EnvironmentalConditions dataclass."""
    env = EnvironmentalConditions(temperature=25, humidity=60)
    assert env.sound_speed == pytest.approx(331.4 + 0.6 * 25)
    # Just checking if it runs without error
    assert isinstance(env.air_absorption_coefficient(), float)


# Tests for room_generator.py

def test_room_generator_init():
    """Test RoomGenerator initialization."""
    generator = RoomGenerator()
    assert generator.generated_rooms == {}


def test_calculate_room_dimensions():
    """Test RoomGenerator's dimension calculation."""
    generator = RoomGenerator()
    dims = generator.calculate_room_dimensions(9.5)  # 9.5m² is a supported size
    assert isinstance(dims, Dimensions3D)
    assert dims.floor_area == pytest.approx(9.5)
    assert dims.height == 3.0

    with pytest.raises(ValueError, match="Unsupported room size"):
        generator.calculate_room_dimensions(100)  # Not in ROOM_ASPECT_RATIOS


def test_room_generator_generate():
    """Test RoomGenerator's generate method."""
    generator = RoomGenerator()

    # Test OFFICE room
    office_room = generator.generate(RoomRole.OFFICE)
    assert isinstance(office_room, Room)
    assert office_room.role == RoomRole.OFFICE
    assert office_room.description == "office"
    assert office_room.reverberation_time_ratio == 0.3
    assert office_room.dimensions.floor_area == pytest.approx(ROOM_SIZES[4])

    # Test CONSULTATION room
    consultation_room = generator.generate(RoomRole.CONSULTATION)
    assert isinstance(consultation_room, Room)
    assert consultation_room.role == RoomRole.CONSULTATION
    assert consultation_room.description == "consultation room"
    assert consultation_room.reverberation_time_ratio == 0.5
    assert consultation_room.dimensions.floor_area == pytest.approx(ROOM_SIZES[3])
