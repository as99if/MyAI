import time
import pygame
import pygame.freetype
from src.interface.speech_engine import SpeechEngine
from src.interface.ui.components.audio_visualizer import AudioVisualizer


def test_TTSEngine():
    """Test TTS Engine with pygame UI and AudioVisualizer component."""
    
    # Initialize pygame
    pygame.init()
    pygame.freetype.init()
    
    # Set up display
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("TTS Engine Test with Audio Visualizer")
    
    # Load font
    try:
        font = pygame.freetype.Font(None, 16)
    except:
        font = pygame.freetype.SysFont("Arial", 16)
    
    # Define colors
    colors = {
        'background': (20, 20, 30),
        'primary': (70, 130, 180),
        'secondary': (100, 149, 237),
        'accent': (255, 215, 0),
        'text': (255, 255, 255),
        'error': (220, 20, 60),
        'success': (34, 139, 34)
    }
    
    # Create AudioVisualizer component
    visualizer_rect = pygame.Rect(50, 50, WINDOW_WIDTH - 100, WINDOW_HEIGHT - 100)
    audio_visualizer = AudioVisualizer(visualizer_rect, font, colors)
    
    # Initialize speech engine
    speech_engine = SpeechEngine(debug=True)
    tts_engine = speech_engine.tts_engine
    
    # Set up callbacks for the TTS engine
    def on_audio_data(audio_data, sample_rate, total_duration):
        """Callback when audio data is available."""
        audio_visualizer.set_audio_data(audio_data, sample_rate, total_duration)
        print(f"Audio data received: {len(audio_data)} samples, {sample_rate}Hz, {total_duration:.2f}s")
    
    def on_playback_position(position, is_playing):
        """Callback for playback position updates."""
        audio_visualizer.set_playback_state(is_playing, position)
        if is_playing:
            print(f"Playback position: {position:.2f}s")
    
    def on_text_stream(text_chunk):
        """Callback for text streaming updates."""
        print(f"Text stream: '{text_chunk}'")
    
    # Set callbacks
    tts_engine.set_audio_data_callback(on_audio_data)
    tts_engine.set_playback_position_callback(on_playback_position)
    tts_engine.set_text_stream_callback(on_text_stream)
    
    # Test text
    test_text = (
        "Welcome to the world of speech synthesis! This is a comprehensive demonstration of "
        "text-to-speech technology with real-time audio visualization. You can press the spacebar "
        "at any time to interrupt the playback and see the beautiful spectrum analyzer in action."
    )
    
    # Initial visualizer state
    audio_visualizer.set_text_content("", test_text)
    
    # Auto-start playback
    print("🎵 Starting TTS Engine Test with Audio Visualizer (auto-start)...")
    print("Press SPACEBAR to interrupt playback")
    print("Press ESC to exit")
    audio_visualizer.set_info_message("🎤 Playback started automatically - Press SPACEBAR to interrupt")
    tts_generator = speech_engine.speak(
        test_text,
        stream=True,
        visualize=True,
        mute=False
    )
    tts_started = True
    spoken_accumulator = ""
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if tts_started:
                        # Interrupt current playback
                        print("Interrupting playback...")
                        tts_engine.stop_playback()
                        audio_visualizer.set_info_message("⏹ Interrupted - Press ESC to quit")
                        tts_started = False
                        tts_generator = None
                    
            
            audio_visualizer.handle_event(event)
        
        # Update streaming
        if tts_started and tts_generator:
            try:
                chunk = next(tts_generator)
                spoken_accumulator += chunk
                remaining = test_text[len(spoken_accumulator):]
                audio_visualizer.set_text_content(spoken_accumulator, remaining)
            except StopIteration:
                print("Playback completed")
                audio_visualizer.set_info_message("✅ Completed - Press SPACE to replay")
                audio_visualizer.set_text_content(test_text, "")
                tts_started = False
                tts_generator = None
            except Exception as e:
                print(f"TTS streaming error: {e}")
                audio_visualizer.set_info_message(f"❌ Error: {e}")
                tts_started = False
                tts_generator = None
        
        # Update visualizer
        audio_visualizer.update()
        
        # Draw
        screen.fill(colors['background'])
        
        title_text = "TTS Engine Test with Audio Visualizer"
        title_rect = font.get_rect(title_text)
        font.render_to(screen, (WINDOW_WIDTH // 2 - title_rect.width // 2, 10), title_text, colors['accent'])
        
        if tts_started:
            instruction_text = "TTS playing... Press SPACEBAR to interrupt | Press ESC to exit"
        else:
            instruction_text = "Press SPACE to start TTS playback | Press ESC to exit"
        instruction_rect = font.get_rect(instruction_text)
        font.render_to(screen, (WINDOW_WIDTH // 2 - instruction_rect.width // 2, 30), instruction_text, colors['text'])
        
        audio_visualizer.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    # Cleanup
    print("Shutting down...")
    if tts_started:
        tts_engine.stop_playback()
    speech_engine.close()
    pygame.quit()
    print("Test completed successfully!")



def test_ASREngine():
    speech_engine = SpeechEngine(debug=True)
    asr_engine = speech_engine.asr_engine
    
    import keyboard
    
    print("Hold SPACEBAR to start recording, release to stop...")
    
    while True:
        if keyboard.is_pressed('space'):
            print("🎙️ Recording... (release spacebar to stop)")
            
            # Start recording while spacebar is held
            audio_data = []
            while keyboard.is_pressed('space'):
                # Record audio chunk while spacebar is pressed
                # This assumes your ASR engine has a method to record in chunks
                chunk = asr_engine.record_chunk()  # You may need to implement this
                if chunk:
                    audio_data.append(chunk)
            
            print("⏹️ Recording stopped. Processing...")
            
            # Process the recorded audio
            if audio_data:
                transcribed_text = asr_engine.transcribe_audio_data(audio_data)  # You may need to implement this
                if transcribed_text:
                    print(f"Transcribed Text: {transcribed_text}")
                else:
                    print("No speech detected or transcription failed.")
            
            print("Hold SPACEBAR to record again, or Ctrl+C to exit...")
        
        time.sleep(0.01)  # Small delay to prevent high CPU usage


if __name__ == "__main__":
    test_TTSEngine()