
from sdialog.audio.room import RoomRole
from sdialog.audio.room_generator import RoomGenerator


def test_generator_simple():
    print(" Room Generator creates:")
    generator = RoomGenerator()
    room = generator.generate(RoomRole.CONSULTATION)
    print(f"  Room {room}")
