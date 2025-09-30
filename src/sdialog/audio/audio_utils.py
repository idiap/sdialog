import re


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
