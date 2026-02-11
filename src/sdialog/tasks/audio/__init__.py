from .spoken_question_answering import SpokenQuestionAnsweringTask
from .automatic_speech_recognition import AutomaticSpeechRecognitionTask
from .spoken_language_understanding import SpokenLanguageUnderstandingTask
from .diarization import DiarizationTask
from .diarization_enhanced import DiarizationEnhancedTask
from .speaker_identification import SpeakerIdentificationTask

__all__ = [
    "SpokenQuestionAnsweringTask",
    "AutomaticSpeechRecognitionTask",
    "SpokenLanguageUnderstandingTask",
    "DiarizationTask",
    "SpeakerIdentificationTask",
    "DiarizationEnhancedTask"
]
