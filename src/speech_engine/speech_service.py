"""SpeechService: Integrated Text-to-Speech and Speech Recognition Engine.

This service provides comprehensive speech processing capabilities:
- Text-to-Speech synthesis using Kokoro
- Speech Recognition using Whisper CPP
- Real-time audio processing
- Voice customization and management
- Multi-language support

Key Components:
-------------
TTSEngine:
- Model: Kokoro ONNX-based neural TTS
- Voice Bank: Pre-trained voice models
- Stream Processing: Real-time audio synthesis

ASREngine:
- Model: Whisper CPP for speech recognition
- Language Detection: Automatic language identification
- Transcription: Real-time and batch processing

Usage Examples:
-------------
1. Text-to-Speech:
    ```python
    tts = TTSEngine()
    audio = tts.engine.create(text="Hello world", voice="en-us-1")
    soundfile.write("output.wav", audio[0], audio[1])
    ```

2. Speech Recognition:
    ```python
    asr = ASREngine()
    segments = asr.engine.transcribe("input.wav")
    for segment in segments:
        print(f"{segment.text}")
    ```

Configuration:
-------------
TTSEngine:
- Model Path: ./models/kokoro-quant.onnx
- Voices Path: ./models/voices.bin
- Supported Languages: 
- Speed Range: 0.5 to 2.0

ASREngine:
- Model Type: medium.en (configurable)
- Models Directory: ./models/
- Language Support: Auto-detection available
- Real-time Processing: Optional
        

Error Handling:
-------------
- Raises ModelNotFoundError: When TTS/ASR models not found
- Raises AudioProcessingError: For audio processing failures
- Raises LanguageNotSupportedError: For unsupported languages
- Raises VoiceNotFoundError: When requested voice unavailable

Returns:
-------
TTSEngine:
- create(text, voice, speed, lang)
- tts.engine.get_voices()
- get_languages()

ASREngine:
- transcribe(audio_file)
- system_info()
- get_params()
- available_languages()
- lang_code, confidence = auto_detect_language(audio_file)

@author - Aisf Ahmed - asif.shuvo2199@outlook.com
"""


import multiprocessing
import os
import soundfile as sf
from kokoro_onnx import Kokoro
from pywhispercpp.model import Model as Whisper_ccp
from mimic3_tts import Mimic3TextToSpeechSystem, Mimic3Settings
# from pathlib import Path


class TTSEngine:
    def __init__(self):
        """
        
        ### Initialization
        - `__init__(model_path: str, voices_path: str, espeak_config: EspeakConfig | None = None)`
        - Creates Kokoro instance with model and voice paths
        - Sets up ONNX runtime session
        - Initializes tokenizer

        ### Main Methods
        1. `create(text: str, voice: str | NDArray[np.float32], speed: float = 1.0, lang: str = "en-us", phonemes: str | None = None, trim: bool = True)`
        - Converts text to audio using specified voice and settings
        - Returns: tuple[NDArray[np.float32], int] (audio data and sample rate)

        2. `create_stream(text: str, voice: str | NDArray[np.float32], speed: float = 1.0, lang: str = "en-us")`
        - Async method that streams audio creation
        - Yields chunks of audio as they're processed
        - Returns: AsyncGenerator[tuple[NDArray[np.float32], int], None]

        ### Utility Methods
        - `get_voices()`: Returns list of available voices
        - `get_languages()`: Returns list of supported languages
        - `get_voice_style(name: str)`: Gets voice style from name

        ### Internal Methods
        - `_create_audio()`: Creates audio from phonemes
        - `_split_phonemes()`: Splits text into processable chunks

        ### Constraints
        - Supported speeds: 0.5 to 2.0
        - Maximum phoneme length enforced
        - Supported languages defined in SUPPORTED_LANGUAGES
        
        """
        
        self.engine = Kokoro("./models/kokoro-quant.onnx",
                             "./models/voices.bin")
        

class ASREngine:
    def __init__(self):
        """
        
        ### 1. Segment Class
        - Represents a transcription segment
        - Properties:
        - t0: start time
        - t1: end time
        - text: transcribed text

        ### 2. Model Class
        Primary interface for transcription operations

        #### Key Methods
        1. `__init__(model: str = 'tiny', models_dir: str = None)`
        - Initializes model with specified model type
        - Downloads model if not present locally

        2. `transcribe(media: Union[str, np.ndarray], n_processors: int = None)`
        - Main transcription method
        - Accepts audio file or numpy array
        - Returns List[Segment]

        3. `auto_detect_language(media, offset_ms: int = 0, n_threads: int = 4)`
        - Detects audio language
        - Returns ((detected_lang, probability), language_probabilities)

        #### Utility Methods
        - `available_languages()`: Lists supported languages
        - `get_params()`: Returns current parameters
        - `system_info()`: Prints system information
        - `print_timings()`: Shows processing timings

        #### Usage Example
        ```python
        # Basic transcription
        model = Model('base.en')
        segments = model.transcribe('audio.mp3')
        for segment in segments:
            print(f"{segment.t0}-{segment.t1}: {segment.text}")

        # Language detection
        model = Model('base')
        lang, probs = model.auto_detect_language('audio.mp3')
        
        """
        
        self.engine = Whisper_ccp(
            model='medium.en',    # or 'large-v3-turbo-q5_0', 'tiny', 'medium-q5_0'
            models_dir="./models/",
            print_realtime=False,
            print_progress=False
        )


def test(tts, asr):

    # Test 1: TTS - Different voices and speeds
    test_texts = {
        "normal": "I've seen angels fall from blinding heights. But you yourself are nothing so divine. Just line",
        "question": "You can't deny the prize it may never fulfill you. It longs to kill you, are you willing to die?",
        "excited": "Until that day. Until the world falls away. Until you say there'll be no more goodbyes. I see it in your eyes Tomorrow never dies!",
    }
    
    for test_name, text in test_texts.items():
        # Test different speeds
        for speed in [0.8, 1.0, 1.2]:
            samples, sample_rate = tts.engine.create(
                text,
                voice="am_adam",
                speed=speed,
                lang="en-us"
            )
            sf.write(f"test_{test_name}_speed_{speed}.wav", samples, sample_rate)
            print(f"Created test_{test_name}_speed_{speed}.wav")

    # Test 2: TTS - Different languages
    multilang_test = {
        "en-us": "Hello, how are you?",
        "fr-fr": "Comment allez-vous?",
    }
    
    for lang, text in multilang_test.items():
        samples, sample_rate = tts.engine.create(
            text,
            voice="am_adam",
            lang=lang
        )
        sf.write(f"test_lang_{lang}.wav", samples, sample_rate)
        print(f"Created test_lang_{lang}.wav")

    # Test 3: ASR - Basic transcription and language detection
    for audio_file in [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.wav')]:
        print(f"\nTesting ASR on {audio_file}")
        
        # Test language detection
        lang_result, lang_probs = asr.engine.auto_detect_language(audio_file)
        print(f"Detected language: {lang_result[0]} (confidence: {lang_result[1]:.2f})")
        
        # Test transcription
        segments = asr.engine.transcribe(audio_file)
        print("Transcription:")
        for seg in segments:
            print(f"{seg.t0/1000:.2f}s -> {seg.t1/1000:.2f}s: {seg.text}")

    # Test 4: ASR - System info and parameters
    print("\nSystem Information:")
    asr.engine.system_info()
    print("\nModel Parameters:")
    print(asr.engine.get_params())
    
    # Test 5: Available voices and languages
    print("\nAvailable TTS voices:", tts.engine.get_voices())
    print("Available TTS languages:", tts.engine.get_languages())
    print("Available ASR languages:", asr.engine.available_languages())

    # Cleanup test files
    for f in [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.wav')]:
        os.remove(f)
    
    
    
    """
    # test asr
    # voices = tts.engine.get_voice_style()
    # print(voices)

    samples, sample_rate = tts.engine.create(
        "And here we go again, we know the start, we know the end Masters of the scene We've done it all before and now we're back to get some more You know what I mean",
        voice="am_adam", speed=1.0
    )
    sf.write(f"meh.wav", samples, sample_rate)

    print(f"Created mehmeh.wav")

    # test tts
    segments = asr.engine.transcribe('meh.wav', new_segment_callback=print)
    print(segments)
    """

def __run__(args):
    tts = TTSEngine()
    asr = ASREngine()
    # test(tts, asr)
    return tts, asr

def run():
    computer_audio = multiprocessing.Process(target=__run__, args=(None, None), name="computer-audio")
    computer_audio.start()
    computer_audio.join()
    