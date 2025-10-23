import pytest
from sdialog.audio.room import Position3D, Dimensions3D, DirectivityType, Room
from sdialog.audio.voice_database import Voice, BaseVoiceDatabase, LocalVoiceDatabase
from sdialog.audio.utils import Role
import os
import shutil


def test_position3d_initialization():
    pos = Position3D(1.0, 2.0, 3.0)
    assert pos.x == 1.0
    assert pos.y == 2.0
    assert pos.z == 3.0


def test_position3d_negative_coords():
    with pytest.raises(ValueError):
        Position3D(-1.0, 2.0, 3.0)


def test_position3d_to_array():
    pos = Position3D(1.0, 2.0, 3.0)
    arr = pos.to_array()
    assert arr.shape == (3,)
    assert all(arr == [1.0, 2.0, 3.0])


def test_position3d_to_list():
    pos = Position3D(1.0, 2.0, 3.0)
    assert pos.to_list() == [1.0, 2.0, 3.0]


def test_position3d_distance_to():
    pos1 = Position3D(0.0, 0.0, 0.0)
    pos2 = Position3D(3.0, 4.0, 0.0)
    assert pos1.distance_to(pos2, dimensions=2) == 5.0
    pos3 = Position3D(3.0, 4.0, 12.0)
    assert pos1.distance_to(pos3, dimensions=3) == 13.0
    with pytest.raises(ValueError):
        pos1.distance_to(pos2, dimensions=4)


def test_position3d_from_list():
    pos = Position3D.from_list([1.0, 2.0, 3.0])
    assert pos.x == 1.0
    assert pos.y == 2.0
    assert pos.z == 3.0
    with pytest.raises(ValueError):
        Position3D.from_list([1.0, 2.0])


def test_dimensions3d_initialization():
    dims = Dimensions3D(width=5.0, length=4.0, height=3.0)
    assert dims.width == 5.0
    assert dims.length == 4.0
    assert dims.height == 3.0


def test_dimensions3d_non_positive_dims():
    with pytest.raises(ValueError):
        Dimensions3D(width=0, length=4.0, height=3.0)
    with pytest.raises(ValueError):
        Dimensions3D(width=5.0, length=-4.0, height=3.0)


def test_dimensions3d_volume():
    dims = Dimensions3D(width=5.0, length=4.0, height=3.0)
    assert dims.volume == 60.0


def test_dimensions3d_to_list():
    dims = Dimensions3D(width=5.0, length=4.0, height=3.0)
    assert dims.to_list() == [5.0, 4.0, 3.0]


@pytest.fixture
def basic_room():
    """Returns a basic Room instance for testing."""
    return Room(dimensions=Dimensions3D(width=10, length=8, height=3))


def test_room_initialization(basic_room):
    assert basic_room.dimensions.width == 10
    assert basic_room.dimensions.length == 8
    assert basic_room.dimensions.height == 3
    assert "center" in basic_room.furnitures
    # Check default speaker placements from model_post_init
    assert "speaker_1" in basic_room.speakers_positions
    assert "speaker_2" in basic_room.speakers_positions
    assert basic_room.mic_position_3d is not None
    assert basic_room.microphone_directivity is not None


def test_place_speaker(basic_room):
    new_speaker_pos = Position3D(2, 2, 1.5)
    basic_room.place_speaker(Role.SPEAKER_1, new_speaker_pos)
    assert Role.SPEAKER_1 in basic_room.speakers_positions
    assert basic_room.speakers_positions[Role.SPEAKER_1] == new_speaker_pos


def test_place_speaker_invalid_name(basic_room):
    invalid_pos = Position3D(2, 2, 1.5)
    with pytest.raises(ValueError):
        basic_room.place_speaker("speaker_4", invalid_pos)


def test_place_speaker_invalid_position(basic_room):
    invalid_pos = Position3D(11, 2, 1.5)  # x > width
    with pytest.raises(ValueError):
        basic_room.place_speaker(Role.SPEAKER_1, invalid_pos)


def test_set_directivity(basic_room):
    basic_room.set_directivity(DirectivityType.NORTH)
    assert basic_room.directivity_type == DirectivityType.NORTH
    assert basic_room.microphone_directivity.azimuth == 0
    assert basic_room.microphone_directivity.colatitude == 90


def test_get_speaker_distances(basic_room):
    distances = basic_room.get_speaker_distances_to_microphone()
    assert "speaker_1" in distances
    assert "speaker_2" in distances
    assert isinstance(distances["speaker_1"], float)
    assert isinstance(distances["speaker_2"], float)


def test_room_to_image(basic_room):
    try:
        from PIL import Image
        img = basic_room.to_image()
        assert isinstance(img, Image.Image)
    except ImportError:
        pytest.skip("Pillow is not installed, skipping image test")


def test_voice_initialization():
    voice = Voice(
        gender="male",
        age=30,
        identifier="v1",
        voice="path/to/v1.wav",
        language="english",
        language_code="en"
    )
    assert voice.gender == "male"
    assert voice.age == 30
    assert voice.identifier == "v1"
    assert voice.voice == "path/to/v1.wav"
    assert voice.language == "english"
    assert voice.language_code == "en"


@pytest.fixture
def sample_voice_data():
    return [
        {"gender": "male", "age": 30, "identifier": "p225", "voice": "p225.wav",
         "language": "english", "language_code": "en"},
        {"gender": "female", "age": 25, "identifier": "p226", "voice": "p226.wav",
         "language": "english", "language_code": "en"},
        {"gender": "male", "age": 45, "identifier": "p227", "voice": "p227.wav",
         "language": "english", "language_code": "en"},
    ]


class MockVoiceDatabase(BaseVoiceDatabase):
    def __init__(self, data):
        self._input_data = data
        super().__init__()

    def populate(self):
        for item in self._input_data:
            self.add_voice(
                gender=item["gender"],
                age=item["age"],
                identifier=item["identifier"],
                voice=item["voice"],
                lang=item["language"],
                language_code=item["language_code"]
            )


def test_base_voice_database_get_voice(sample_voice_data):
    db = MockVoiceDatabase(sample_voice_data)
    voice = db.get_voice(gender="female", age=26, lang="english")
    assert voice.gender == "female"
    assert voice.age == 25  # Closest age


def test_base_voice_database_no_duplicates(sample_voice_data):
    db = MockVoiceDatabase(sample_voice_data)
    voice1 = db.get_voice(gender="male", age=30, lang="english", keep_duplicate=False)
    voice2 = db.get_voice(gender="male", age=45, lang="english", keep_duplicate=False)
    assert voice1.identifier != voice2.identifier

    with pytest.raises(ValueError):
        db.get_voice(gender="male", age=30, lang="english", keep_duplicate=False)


@pytest.fixture(scope="module")
def local_voice_db_setup():
    temp_dir = "tests/data/temp_voices_for_test"
    os.makedirs(temp_dir, exist_ok=True)

    # Create dummy audio files
    with open(os.path.join(temp_dir, "yanis.wav"), "w") as f:
        f.write("dummy")
    with open(os.path.join(temp_dir, "thomas.wav"), "w") as f:
        f.write("dummy")

    # Create metadata file
    metadata_path = os.path.join(temp_dir, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("identifier,gender,age,file_name,language,language_code\n")
        f.write("yanis,male,30,yanis.wav,french,fr\n")
        f.write("thomas,male,25,thomas.wav,english,en\n")

    yield temp_dir, metadata_path

    shutil.rmtree(temp_dir)


def test_local_voice_database(local_voice_db_setup):
    audio_dir, metadata_file = local_voice_db_setup
    db = LocalVoiceDatabase(directory_audios=audio_dir, metadata_file=metadata_file)

    assert "french" in db.get_data()
    assert "english" in db.get_data()

    voice = db.get_voice("male", 32, "french")
    assert voice.identifier == "yanis"
