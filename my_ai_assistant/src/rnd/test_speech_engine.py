import wave
import io
import pygame
from piper import PiperVoice, SynthesisConfig

voice = PiperVoice.load("/Users/asifahmed/personal/development/MyAI/my_ai_assistant/models/speech/en_GB-alba-medium.onnx")

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.0,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

def play_synthesized_audio(text, voice, syn_config):
    """
    Synthesize and play audio directly without saving to file
    """
    # Create an in-memory bytes buffer
    audio_buffer = io.BytesIO()
    
    # Synthesize audio to the buffer
    with wave.open(audio_buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file=wav_file, syn_config=syn_config)
    
    # Reset buffer position to beginning
    audio_buffer.seek(0)
    
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load and play the audio from buffer
    pygame.mixer.music.load(audio_buffer)
    pygame.mixer.music.play()
    
    # Wait for playback to complete
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    
    pygame.mixer.quit()

print('meh')

text = "Welcome to the world of speech synthesis!"
play_synthesized_audio(text, voice, syn_config)