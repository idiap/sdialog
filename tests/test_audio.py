import os
import pytest
import numpy as np

from sdialog.audio.voice_database import Voice, is_a_audio_file
from sdialog.audio.voice_database import BaseVoiceDatabase, LocalVoiceDatabase, VoiceDatabase
from sdialog.audio.tts import BaseTTS
from unittest.mock import MagicMock, patch


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
def local_voice_db_setup(tmp_path_factory):
    temp_dir = str(tmp_path_factory.mktemp("voices"))

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

    return temp_dir, metadata_path


def test_local_voice_database_setup(local_voice_db_setup):
    audio_dir, metadata_file = local_voice_db_setup
    db = LocalVoiceDatabase(directory_audios=audio_dir, metadata_file=metadata_file)

    assert "french" in db.get_data()
    assert "english" in db.get_data()

    voice = db.get_voice("male", 32, "french")
    assert voice.identifier == "yanis"


class MockTTS(BaseTTS):
    """A mock TTS engine for testing purposes."""
    def generate(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        """Generates a dummy audio signal."""
        return (np.zeros(16000), 16000)


@pytest.fixture
def mock_tts():
    """Returns a MockTTS instance for testing."""
    return MockTTS()


def test_tts_initialization(mock_tts):
    """Tests the initialization of the mock TTS engine."""
    assert isinstance(mock_tts, BaseTTS)
    assert mock_tts.pipeline is None


def test_tts_generate(mock_tts):
    """Tests the audio generation of the mock TTS engine."""
    audio, sr = mock_tts.generate("hello", "voice1")
    assert isinstance(audio, np.ndarray)
    assert isinstance(sr, int)
    assert sr == 16000
    assert audio.shape == (16000,)


def test_base_tts_abstract():
    """Tests that BaseTTS cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTTS()


def test_is_a_audio_file():
    """Tests the audio file extension checker."""
    assert is_a_audio_file("test.wav")
    assert is_a_audio_file("hello.MP3")
    assert not is_a_audio_file("document.txt")
    assert not is_a_audio_file("archive.zip")


@pytest.fixture
def in_memory_db():
    """Returns an in-memory VoiceDatabase for testing."""
    data = [
        {"gender": "m", "age": 25, "identifier": "id1", "voice": "voice1",
         "language": "english", "language_code": "en"},
        {"gender": "female", "age": 30, "identifier": "id2", "voice": "voice2",
         "language": "english", "language_code": "en"},
        {"gender": "male", "age": 25, "identifier": "id3", "voice": "voice3",
         "language": "french", "language_code": "fr"},
    ]
    return VoiceDatabase(data)


def test_voice_database_gender_conversion(in_memory_db):
    """Tests the internal _gender_to_gender method."""
    assert in_memory_db._gender_to_gender("m") == "male"
    assert in_memory_db._gender_to_gender("F") == "female"
    with pytest.raises(ValueError):
        in_memory_db._gender_to_gender("unknown")


def test_voice_database_get_by_identifier(in_memory_db):
    """Tests retrieving a voice by its identifier."""
    voice = in_memory_db.get_voice_by_identifier("id1", "english")
    assert voice.identifier == "id1"

    with pytest.raises(ValueError, match="not found in the database"):
        in_memory_db.get_voice_by_identifier("nonexistent", "english")

    with pytest.raises(ValueError, match="Language englishs not found"):
        in_memory_db.get_voice_by_identifier("id1", "englishs")


def test_voice_database_reset_used_voices(in_memory_db):
    """Tests the reset functionality for used voices."""
    # Use a voice
    in_memory_db.get_voice("male", 25, "english", keep_duplicate=False)
    assert "english" in in_memory_db._used_voices
    assert "id1" in in_memory_db._used_voices["english"]

    # Reset
    in_memory_db.reset_used_voices()
    assert in_memory_db._used_voices == {}


def test_voice_database_get_statistics(in_memory_db):
    """Tests the statistics generation."""
    stats_dict = in_memory_db.get_statistics()
    assert stats_dict["num_languages"] == 2
    assert stats_dict["overall"]["total"] == 3

    stats_pretty = in_memory_db.get_statistics(pretty=True)
    assert isinstance(stats_pretty, str)
    assert "Voice Database Statistics" in stats_pretty
    assert "english" in stats_pretty


def test_voice_database_populate_errors():
    """Tests error handling in VoiceDatabase populate method."""
    with pytest.raises(ValueError, match="is not a list of dictionaries"):
        VoiceDatabase("not a list")

    with pytest.raises(ValueError, match="Voice column does not exist"):
        VoiceDatabase([{"gender": "m", "age": 25, "identifier": "id1",
                        "language": "english", "language_code": "en"}])


def test_huggingface_voice_database_populate_with_mock():
    """Tests the HuggingfaceVoiceDatabase with a mocked datasets module."""
    from sdialog.audio.voice_database import HuggingfaceVoiceDatabase

    mock_dataset_content = [
        # Entry with full metadata
        {"gender": "f", "age": 45, "identifier": "hf1", "audio": {"path": "p1.wav"},
         "language": "german", "language_code": "de"},
        # Entry with missing optional fields
        {"gender": "male", "age": 50, "identifier": "hf2", "voice": "voice_hf2"},
        # Entry with missing mandatory fields that should be randomized
        {"identifier": "hf3", "voice": "voice_hf3"},
    ]

    # Create a mock for the 'datasets' module
    mock_datasets_module = MagicMock()
    mock_datasets_module.load_dataset.return_value = {"train": mock_dataset_content}

    # Use patch.dict to temporarily replace the 'datasets' module in sys.modules
    with patch.dict('sys.modules', {'datasets': mock_datasets_module}):
        db = HuggingfaceVoiceDatabase("fake/dataset")

        # Verify population
        assert len(db.get_data()) > 0
        assert "german" in db.get_data()
        assert "english" in db.get_data()  # Default language

        # Check if a specific voice was added correctly
        voice = db.get_voice_by_identifier("hf1", "german")
        assert voice.age == 45
        assert voice.language_code == "de"

        # Check that random values were filled in
        voice3 = db.get_voice_by_identifier("hf3", "english")
        assert isinstance(voice3.age, int)
        assert voice3.gender in ["male", "female"]


def test_local_voice_database_linter(tmp_path):
    """Tests the LocalVoiceDatabase with different metadata files."""
    from sdialog.audio.voice_database import LocalVoiceDatabase

    audio_dir = tmp_path / "audios"
    audio_dir.mkdir()
    (audio_dir / "voice1.wav").touch()

    # Test with CSV
    csv_file = tmp_path / "metadata.csv"
    csv_file.write_text("identifier,gender,age,file_name,language\nid1,male,30,voice1.wav,english")
    db_csv = LocalVoiceDatabase(str(audio_dir), str(csv_file))
    assert db_csv.get_voice_by_identifier("id1", "english").age == 30

    # Test with JSON
    json_file = tmp_path / "metadata.json"
    json_content = ('[{"identifier": "id2", "gender": "f", "age": 40, '
                    '"voice": "id2_voice", "language": "french"}]')
    json_file.write_text(json_content)
    db_json = LocalVoiceDatabase(str(audio_dir), str(json_file))
    assert db_json.get_voice_by_identifier("id2", "french").age == 40

    # Test error handling
    with pytest.raises(ValueError, match="Directory audios does not exist"):
        LocalVoiceDatabase("nonexistent_dir", str(csv_file))

    with pytest.raises(ValueError, match="Metadata file does not exist"):
        LocalVoiceDatabase(str(audio_dir), "nonexistent.csv")

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("id,sex,years")  # Missing required columns
    with pytest.raises(ValueError, match="Voice or file_name column does not exist"):
        LocalVoiceDatabase(str(audio_dir), str(bad_csv))
