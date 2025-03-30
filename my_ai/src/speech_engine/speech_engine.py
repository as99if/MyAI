"""
@author - Aisf Ahmed - asif.shuvo2199@outlook.com
"""


import multiprocessing
import os
# import soundfile as sf
from kokoro_onnx import Kokoro
from pywhispercpp.model import Model as Whisper_ccp
# import torch
from pydub import AudioSegment
# import whisper
# from pathlib import Path


class TTSEngine:
    def __init__(self):
        
        
        self.engine = Kokoro("./models/speech/kokoro-quant.onnx",
                             "./models/speech/voices.bin")
        # download from here - https://github.com/thewh1teagle/kokoro-onnx?tab=readme-ov-file#setup
        

class ASREngine:
    def __init__(self):
        
        # self.engine = whisper.load_model("medium.en", download_root="./models/speech/", in_memory=False)
        self.engine = Whisper_ccp(
            model="large-v3-turbo", # large-v3, large-v3-turbo-q5_0, large-v3-turbo-q5_0, medium.en, large-v3
            models_dir="./models/speech/",
            print_realtime=False,
            print_progress=False
        )
        

    

class SpeechEngine:
    
    def __init__(self):
        self.tts_engine = TTSEngine()
        self.asr_engine = ASREngine()


# def test(tts, asr):

#     # Test 1: TTS - Different voices and speeds
#     test_texts = {
#         "normal": "I've seen angels fall from blinding heights. But you yourself are nothing so divine. Just line",
#         "question": "You can't deny the prize it may never fulfill you. It longs to kill you, are you willing to die?",
#         "excited": "Until that day. Until the world falls away. Until you say there'll be no more goodbyes. I see it in your eyes Tomorrow never dies!",
#     }
    
#     for test_name, text in test_texts.items():
#         # Test different speeds
#         for speed in [0.8, 1.0, 1.2]:
#             samples, sample_rate = tts.engine.create(
#                 text,
#                 voice="am_adam",
#                 speed=speed,
#                 lang="en-us"
#             )
#             sf.write(f"test_{test_name}_speed_{speed}.wav", samples, sample_rate)
#             print(f"Created test_{test_name}_speed_{speed}.wav")

#     # Test 2: TTS - Different languages
#     multilang_test = {
#         "en-us": "Hello, how are you?",
#         "fr-fr": "Comment allez-vous?",
#     }
    
#     for lang, text in multilang_test.items():
#         samples, sample_rate = tts.engine.create(
#             text,
#             voice="am_adam",
#             lang=lang
#         )
#         sf.write(f"test_lang_{lang}.wav", samples, sample_rate)
#         print(f"Created test_lang_{lang}.wav")

#     # Test 3: ASR - Basic transcription and language detection
#     for audio_file in [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.wav')]:
#         print(f"\nTesting ASR on {audio_file}")
        
#         # Test language detection
#         lang_result, lang_probs = asr.engine.auto_detect_language(audio_file)
#         print(f"Detected language: {lang_result[0]} (confidence: {lang_result[1]:.2f})")
        
#         # Test transcription
#         segments = asr.engine.transcribe(audio_file)
#         print("Transcription:")
#         for seg in segments:
#             print(f"{seg.t0/1000:.2f}s -> {seg.t1/1000:.2f}s: {seg.text}")

#     # Test 4: ASR - System info and parameters
#     print("\nSystem Information:")
#     asr.engine.system_info()
#     print("\nModel Parameters:")
#     print(asr.engine.get_params())
    
#     # Test 5: Available voices and languages
#     print("\nAvailable TTS voices:", tts.engine.get_voices())
#     print("Available TTS languages:", tts.engine.get_languages())
#     print("Available ASR languages:", asr.engine.available_languages())

#     # Cleanup test files
#     for f in [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.wav')]:
#         os.remove(f)
    
    
    
#     """
#     # test asr
#     # voices = tts.engine.get_voice_style()
#     # print(voices)

#     samples, sample_rate = tts.engine.create(
#         "And here we go again, we know the start, we know the end Masters of the scene We've done it all before and now we're back to get some more You know what I mean",
#         voice="am_adam", speed=1.0
#     )
#     sf.write(f"meh.wav", samples, sample_rate)

#     print(f"Created mehmeh.wav")

#     # test tts
#     segments = asr.engine.transcribe('meh.wav', new_segment_callback=print)
#     print(segments)
#     """

# def __run__(args):
#     tts = TTSEngine()
#     asr = ASREngine()
#     # test(tts, asr)
#     return tts, asr

# def run():
#     computer_audio = multiprocessing.Process(target=__run__, args=(None, None), name="computer-audio")
#     computer_audio.start()
#     computer_audio.join()