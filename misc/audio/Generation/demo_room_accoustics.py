from sdialog.audio.room_generator import RoomGenerator

room_generator = RoomGenerator()

room = room_generator.generate("office")

print(room)
