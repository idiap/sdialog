import re
import torch


class AudioUtils:
    """
    Audio utilities.
    """

    whisper_model = {}

    @staticmethod
    def get_whisper_model(model_name: str = "large-v3", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Get the whisper model.
        """
        import whisper

        if model_name not in AudioUtils.whisper_model:
            AudioUtils.whisper_model[model_name] = whisper.load_model(model_name, device=device)

        return AudioUtils.whisper_model[model_name]

    @staticmethod
    def remove_audio_tags(text: str) -> str:
        """
        Remove all the tags that use those formatting: <>, {}, (), []
        """
        return re.sub(r'<[^>]*>', '', text).replace("*", "")
