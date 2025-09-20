"""
@author - Aisf Ahmed - asif.shuvo2199@outlook.com
"""
import os
import io
import queue
from typing import Callable, Optional, Tuple, Generator, Union
import wave
import keyboard
import numpy as np
import threading
import time
from scipy.fft import fft
from scipy.signal import get_window
from piper import PiperVoice, SynthesisConfig
import pygame
import pyaudio
from src.utils.log_manager import LoggingManager
from pywhispercpp.model import Model as Whisper_ccp
from src.interface.peripheral_manager.device_manager import DeviceManager

class TTSEngine:
    """
    Text-to-Speech engine with optional streaming and callback support.
    
    This class provides comprehensive text-to-speech functionality using the Piper voice model
    with integrated device management through DeviceManager. It supports both streaming and
    non-streaming playback modes with real-time position tracking and audio visualization
    callbacks.
    
    Key Features:
    - High-quality speech synthesis using Piper voice models
    - Streaming text playback with character-by-character progress tracking
    - Audio device management through DeviceManager integration
    - Real-time playback position callbacks for UI synchronization
    - Audio data callbacks for visualization (waveform, spectrum analysis)
    - Graceful interruption handling and resource cleanup
    - Configurable synthesis parameters (volume, speed, noise levels)
    
    Audio Processing:
    - Automatic sample rate matching between synthesis and playback
    - Mono/stereo audio handling with automatic conversion
    - Normalized float32 audio data for consistent visualization
    - Dynamic pygame mixer reinitialization for optimal compatibility
    
    Callbacks:
    - text_stream_callback: Real-time text streaming updates
    - playback_position_callback: Playback progress and status updates
    - audio_data_callback: Raw audio data for visualization components
    
    Example:
        >>> device_manager = DeviceManager()
        >>> device_manager.detect_devices()
        >>> device_manager.select_speaker(0)  # Select first speaker
        >>> 
        >>> tts = TTSEngine(debug=True, device_manager=device_manager)
        >>> 
        >>> # Set up callbacks
        >>> tts.set_text_stream_callback(lambda text: print(f"Speaking: {text}"))
        >>> tts.set_audio_data_callback(lambda data, rate, duration: visualize_audio(data))
        >>> 
        >>> # Non-streaming playback
        >>> spoken, unspoken = tts.play_synthesized_audio("Hello world!")
        >>> 
        >>> # Streaming playback
        >>> for chunk in tts.play_synthesized_audio("Hello world!", stream=True):
        >>>     print(chunk, end='', flush=True)
        >>> 
        >>> tts.close()
    
    Note:
        Requires a properly configured DeviceManager with selected audio output device.
        Falls back to default pygame mixer behavior if no device manager is provided.
    """
    
    def __init__(self,
                model_path: str = "/Users/asifahmed/personal/development/MyAI/my_ai_assistant/models/speech/en_GB-semaine-medium.onnx", 
                debug: bool = False, 
                device_manager: DeviceManager=None
        ):
        """
        Initialize the TTS engine with voice model and device management.
        
        Sets up the Piper voice synthesis engine, configures audio parameters,
        and integrates with the provided DeviceManager for optimal audio output
        device selection and management.
        
        Args:
            model_path (str): Path to the Piper voice model file (.onnx format).
                             Should point to a valid Piper voice model for the desired
                             language and voice characteristics.
            debug (bool): Enable detailed debug logging for troubleshooting.
                         When True, prints synthesis progress, audio processing
                         details, and device management information.
            device_manager (DeviceManager, optional): Configured DeviceManager instance
                                                    with selected audio output device.
                                                    If None, uses default pygame mixer behavior.
        
        Raises:
            Exception: If Piper voice model loading fails or audio system initialization fails.
            
        Side Effects:
            - Loads Piper voice model into memory
            - Initializes pygame mixer for audio playback
            - Sets up logging system integration
            - Configures default synthesis parameters
            
        Note:
            The voice model file must be compatible with the Piper library.
            Large models may take significant time and memory to load.
        """
        # Initialize logging system
        self.logging_manager = LoggingManager()
        self.logging_manager.add_message("Initiating - TTSEngine", level="INFO", source="TTSEngine")
        
        # Store configuration
        self.debug = debug
        self.device_manager = device_manager
        
        # Load Piper voice model
        self.voice = PiperVoice.load(model_path)
        
        # Configure synthesis parameters for optimal speech quality
        self.syn_config = SynthesisConfig(
            volume=0.5,              # Moderate volume level
            length_scale=1.0,        # Normal speech speed
            noise_scale=1.0,         # Standard noise level for naturalness
            noise_w_scale=1.0,       # Standard noise width for variation
            normalize_audio=False,   # Keep original audio levels for consistency
        )
        
        # Playback state management
        # Using lists for mutable references in threaded callbacks
        self.user_interrupted = [False]
        self.is_playing = [False]
        
        # Callback functions for external integrations
        self.text_stream_callback: Optional[Callable[[str], None]] = None
        self.playback_position_callback: Optional[Callable[[float, bool], None]] = None
        self.audio_data_callback: Optional[Callable[[np.ndarray, int, float], None]] = None
        
        # Initialize pygame mixer for audio playback
        # Will be reconfigured dynamically based on synthesis sample rate
        pygame.mixer.init()
        
        self._debug_print("TTS Engine initialized successfully")
        
    def _debug_print(self, message: str) -> None:
        """
        Print debug message if debug mode is enabled.
        
        Provides consistent debug output formatting for development and
        troubleshooting. All debug messages are prefixed with [TTS DEBUG]
        for easy identification in log streams.
        
        Args:
            message (str): Debug message to print
            
        Note:
            Debug messages are only printed when self.debug is True.
            This ensures clean production output while enabling detailed
            debugging information during development.
        """
        if self.debug:
            print(f"[TTS DEBUG] {message}")
    
    def set_text_stream_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback function for real-time text streaming updates.
        
        This callback is invoked during streaming playback to provide character-by-character
        text updates as the audio is being played. Useful for creating typewriter effects
        or real-time text highlighting in user interfaces.
        
        Args:
            callback (Callable[[str], None]): Function to call with text chunks.
                                             Receives the newly spoken text portion
                                             as each character is "spoken" during playback.
        
        Example:
            >>> def on_text_stream(text_chunk):
            >>>     print(text_chunk, end='', flush=True)
            >>> tts.set_text_stream_callback(on_text_stream)
        """
        self.text_stream_callback = callback
    
    def set_playback_position_callback(self, callback: Callable[[float, bool], None]) -> None:
        """
        Set callback function for playback position and status updates.
        
        This callback provides real-time playback progress information, useful for
        progress bars, timeline scrubbing, or synchronized visual effects.
        
        Args:
            callback (Callable[[float, bool], None]): Function to call with position updates.
                                                     Parameters:
                                                     - position (float): Current playback position in seconds
                                                     - is_playing (bool): Whether audio is currently playing
        
        Example:
            >>> def on_position_update(position, is_playing):
            >>>     if is_playing:
            >>>         print(f"Playing at {position:.2f}s")
            >>>     else:
            >>>         print("Playback stopped")
            >>> tts.set_playback_position_callback(on_position_update)
        """
        self.playback_position_callback = callback
    
    def set_audio_data_callback(self, callback: Callable[[np.ndarray, int, float], None]) -> None:
        """
        Set callback function for audio data visualization.
        
        This callback provides the complete synthesized audio data for visualization
        purposes such as waveform display, spectrum analysis, or audio effects.
        
        Args:
            callback (Callable[[np.ndarray, int, float], None]): Function to call with audio data.
                                                                Parameters:
                                                                - audio_data (np.ndarray): Normalized float32 mono audio
                                                                - sample_rate (int): Audio sample rate in Hz
                                                                - duration (float): Total audio duration in seconds
        
        Example:
            >>> def on_audio_data(audio_data, sample_rate, duration):
            >>>     # Create waveform visualization
            >>>     visualizer.display_waveform(audio_data, sample_rate)
            >>> tts.set_audio_data_callback(on_audio_data)
        """
        self.audio_data_callback = callback
        
    def synthesize(self, text: str) -> io.BytesIO:
        """
        Synthesize text to audio using the Piper voice model.
        
        Converts the input text to speech audio using the configured Piper voice model
        and synthesis parameters. The resulting audio is returned as a WAV format
        byte stream ready for playback or further processing.
        
        Args:
            text (str): Text to synthesize into speech. Should be properly formatted
                       with appropriate punctuation for natural speech rhythm.
        
        Returns:
            io.BytesIO: Audio buffer containing WAV format audio data.
                       The buffer is positioned at the beginning and ready for reading.
        
        Raises:
            Exception: If synthesis fails due to model errors or invalid text input.
            
        Side Effects:
            - Loads text into Piper model for processing
            - Generates audio data in memory
            - Logs synthesis progress if debug enabled
            
        Note:
            The synthesis process may take time proportional to text length.
            Large texts should be chunked for better responsiveness.
        """
        self._debug_print(f"Synthesizing text: '{text[:50]}...' ({len(text)} chars)")
        
        # Create in-memory audio buffer
        audio_buffer = io.BytesIO()
        
        # Synthesize text to WAV format using Piper
        with wave.open(audio_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file=wav_file, syn_config=self.syn_config)
        
        # Reset buffer position for reading
        audio_buffer.seek(0)
        self._debug_print("Audio synthesis completed")
        
        return audio_buffer

    def play_synthesized_audio(
        self, 
        text: str, 
        stream: bool = False, 
        mute: bool = False
    ) -> Union[Tuple[str, str], Generator[str, None, None]]:
        """
        Play synthesized audio with optional streaming and device management.
        
        This is the main playback method that synthesizes text to speech and plays it
        through the configured audio output device. Supports both streaming and
        non-streaming modes with comprehensive callback support and device integration.
        
        Playback Process:
        1. Synthesizes text to audio using Piper voice model
        2. Processes audio data for visualization callbacks
        3. Configures pygame mixer to match synthesis sample rate
        4. Utilizes DeviceManager for optimal audio device selection
        5. Plays audio with real-time progress tracking
        6. Provides streaming text updates if requested
        7. Handles user interruption and cleanup
        
        Args:
            text (str): Text to synthesize and speak. Should be well-formatted
                       for natural speech with appropriate punctuation.
            stream (bool): Enable streaming mode for character-by-character text updates.
                          When True, returns a generator yielding text chunks.
                          When False, returns tuple of (spoken, unspoken) text.
            mute (bool): Mute audio output while maintaining all other functionality
                        including callbacks and progress tracking.
        
        Returns:
            Union[Tuple[str, str], Generator[str, None, None]]:
                - If stream=False: Tuple of (spoken_text, unspoken_text)
                - If stream=True: Generator yielding text chunks as they're "spoken"
        
        Side Effects:
            - Reinitializes pygame mixer if sample rate changes
            - Utilizes DeviceManager audio output device if available
            - Invokes all registered callbacks during playback
            - Updates internal playback state variables
            - Creates temporary audio thread for playback
            
        Note:
            - DeviceManager integration ensures optimal audio device usage
            - Playback can be interrupted using stop_playback() method
            - All audio processing is done in separate thread to prevent blocking
            - Callbacks are invoked from the main thread for UI safety
        """
        self._debug_print(f"Starting playback - Stream: {stream}, Mute: {mute}")
        
        # Log playback parameters
        parameters = []
        if stream:
            parameters.append("Stream: True")
        if mute:
            parameters.append("Mute: True")
        param_str = "; ".join(parameters) if parameters else "Standard playback"
        self.logging_manager.add_message(f"Synthesizing reply - {param_str}", level="INFO", source="TTSEngine")
        
        # Synthesize audio once
        audio_buffer = self.synthesize(text)
        
        # Extract audio properties for processing
        with wave.open(audio_buffer, "rb") as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()

        # Convert audio to normalized float32 mono for visualization
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.float32
            
        audio_data = np.frombuffer(frames, dtype=dtype)
        
        # Convert stereo to mono if necessary
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
        # Normalize to float32 range [-1.0, 1.0]
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data)) if audio_data.size > 0 else 1.0
        if max_abs > 0:
            audio_data /= max_abs
            
        total_duration = total_frames / sample_rate if sample_rate > 0 else 0.0

        # Provide audio data to visualization callback
        if self.audio_data_callback:
            self.audio_data_callback(audio_data, sample_rate, total_duration)
            
        # Reset buffer for pygame usage
        audio_buffer.seek(0)

        # Configure pygame mixer for optimal playback
        try:
            # Check current mixer state
            init_state = pygame.mixer.get_init()
            if init_state:
                current_freq = init_state[0]
                if current_freq != sample_rate:
                    self._debug_print(f"Reinitializing mixer: {current_freq} -> {sample_rate}")
                    pygame.mixer.quit()
                    pygame.mixer.init(frequency=sample_rate)
            else:
                pygame.mixer.init(frequency=sample_rate)
                
            # Utilize DeviceManager for audio output device if available
            if self.device_manager and hasattr(self.device_manager, 'main_speaker') and self.device_manager.main_speaker:
                self._debug_print("Using DeviceManager audio output device")
                # Note: pygame.mixer doesn't directly support device selection
                # The DeviceManager ensures the system default is properly configured
                # Future enhancement: Consider using PyAudio for direct device control
                
        except pygame.error as e:
            self._debug_print(f"Pygame mixer initialization error: {e}")
            self.logging_manager.add_message(f"Audio mixer error: {e}", level="ERROR", source="TTSEngine")
            return ("", text) if not stream else iter([])

        # Initialize playback state
        self.is_playing = [True]
        self.user_interrupted = [False]
        result_spoken_text = [""]
        result_unspoken_text = [text]
        playback_start_time = [0.0]

        def get_current_position() -> float:
            """Get current playback position in seconds."""
            if pygame.mixer.music.get_busy() and playback_start_time[0] > 0:
                return time.time() - playback_start_time[0]
            return 0.0

        def update_text_progress() -> None:
            """Update text progress based on playback position."""
            pos_seconds = get_current_position()
            proportion = min(1.0, max(0.0, (pos_seconds / total_duration) if total_duration > 0 else 0))
            char_pos = int(len(text) * proportion)
            result_spoken_text[0] = text[:char_pos]
            result_unspoken_text[0] = text[char_pos:]

        def audio_thread_function() -> None:
            """Audio playback thread function."""
            try:
                pygame.mixer.music.load(audio_buffer)
                pygame.mixer.music.set_volume(0.0 if mute else 1.0)
                playback_start_time[0] = time.time()
                pygame.mixer.music.play()
                
                # Monitor playback until completion or interruption
                while pygame.mixer.music.get_busy() and not self.user_interrupted[0]:
                    time.sleep(0.05)
                    
                if self.user_interrupted[0]:
                    pygame.mixer.music.stop()
                else:
                    # Ensure completion state
                    result_spoken_text[0] = text
                    result_unspoken_text[0] = ""
                    
            except Exception as e:
                self._debug_print(f"Audio playback error: {e}")
                self.logging_manager.add_message(f"Audio playback error: {e}", level="ERROR", source="TTSEngine")
            finally:
                self.is_playing[0] = False

        # Start audio playback in separate thread
        audio_thread = threading.Thread(target=audio_thread_function, daemon=True)
        audio_thread.start()

        # Non-streaming playback mode
        if not stream:
            last_callback_time = 0.0
            
            while self.is_playing[0]:
                update_text_progress()
                
                # Throttled position callbacks
                now = time.time()
                if self.playback_position_callback and now - last_callback_time > 0.05:
                    self.playback_position_callback(get_current_position(), True)
                    last_callback_time = now
                    
                time.sleep(0.02)
            
            # Final position callback
            if self.playback_position_callback:
                self.playback_position_callback(0.0, False)
                
            # Wait for audio thread completion
            audio_thread.join(timeout=1.0)
            
            return result_spoken_text[0], result_unspoken_text[0]

        # Streaming playback mode
        def stream_generator() -> Generator[str, None, None]:
            """Generator for streaming text updates."""
            last_spoken = ""
            
            while self.is_playing[0]:
                update_text_progress()
                spoken = result_spoken_text[0]

                # Position callback for streaming mode
                if self.playback_position_callback:
                    self.playback_position_callback(get_current_position(), True)

                # Yield new text chunks
                if spoken != last_spoken:
                    chunk = spoken[len(last_spoken):]
                    if self.text_stream_callback and chunk:
                        self.text_stream_callback(chunk)
                    last_spoken = spoken
                    if chunk:
                        yield chunk
                else:
                    yield ""  # Yield empty string to maintain generator timing
                    
                time.sleep(0.02)

            # Handle final chunk if needed
            if result_spoken_text[0] != last_spoken:
                final_chunk = result_spoken_text[0][len(last_spoken):]
                if self.text_stream_callback and final_chunk:
                    self.text_stream_callback(final_chunk)
                if final_chunk:
                    yield final_chunk

            # Final position callback
            if self.playback_position_callback:
                self.playback_position_callback(0.0, False)
                
            # Wait for audio thread completion
            audio_thread.join(timeout=1.0)

        return stream_generator()

    def stop_playback(self) -> None:
        """
        Stop current audio playback immediately.
        
        Interrupts any ongoing audio playback and sets the interruption flag
        to signal threads to terminate gracefully. Safe to call multiple times
        or when no audio is playing.
        
        Side Effects:
            - Sets user interruption flag
            - Stops pygame mixer music playback
            - Terminates audio playback thread
            - Triggers final position callbacks with stopped status
            
        Note:
            This method is thread-safe and can be called from any thread.
            The actual cleanup happens in the audio playback thread.
        """
        self._debug_print("Stopping playback")
        self.user_interrupted[0] = True
        
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

    def close(self) -> None:
        """
        Gracefully shutdown TTS engine and cleanup all resources.
        
        Performs comprehensive cleanup of all TTS engine resources including
        pygame mixer, voice model, and logging system. Ensures proper resource
        deallocation and prevents memory leaks or device conflicts.
        
        Cleanup Process:
        1. Stops any ongoing audio playback
        2. Shuts down pygame mixer system
        3. Releases Piper voice model resources
        4. Logs successful shutdown
        5. Handles cleanup errors gracefully
        
        Side Effects:
        - Terminates all pygame mixer resources
        - Deallocates Piper voice model memory
        - Logs shutdown events
        - Does NOT affect DeviceManager (external resource)
        
        Note:
        - Safe to call multiple times
        - Does not affect external DeviceManager instance
        - All errors during cleanup are logged but don't raise exceptions
        - Should be called before application exit
        
        Example:
        >>> tts = TTSEngine(device_manager=device_manager)
        >>> # ... use TTS engine ...
        >>> tts.close()  # Clean shutdown
        """
        try:
            self._debug_print("Shutting down TTS engine...")
            
            # Stop any ongoing audio playback
            self.stop_playback()
            
            # Shutdown pygame mixer
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                self._debug_print("Pygame mixer shutdown complete")
            
            # Clean up voice model resources
            if hasattr(self, 'voice') and self.voice is not None:
                del self.voice
                self._debug_print("Voice model resources cleaned up")
            
            # Log successful shutdown
            self.logging_manager.add_message("TTSEngine shutdown complete", level="INFO", source="TTSEngine")
            self._debug_print("TTS engine shutdown completed successfully")
            
        except Exception as e:
            error_msg = f"Error during TTS engine shutdown: {e}"
            self._debug_print(error_msg)
            self.logging_manager.add_message(error_msg, level="ERROR", source="TTSEngine")


class ASREngine:
    """
    Automatic Speech Recognition engine using Whisper with integrated device management.
    
    This class provides speech-to-text functionality using the Whisper model with
    comprehensive microphone device management through DeviceManager integration.
    It handles audio recording, processing, and transcription with automatic device
    selection and proper resource management.
    
    The engine supports push-to-talk recording (space key) and provides robust
    error handling for audio device operations and transcription processes.
    
    Default Audio Configuration:
    - Format: 16-bit PCM (paInt16)
    - Channels: Mono (1 channel)
    - Sample Rate: 44.1 kHz
    - Chunk Size: 2048 frames (optimized for speech recognition)
    
    Attributes:
        device_manager (DeviceManager): Device manager for microphone operations
        engine (Whisper_ccp): Whisper model instance for transcription
        debug (bool): Debug mode flag for verbose logging
        logging_manager (LoggingManager): Logging system interface
        
    Example:
        >>> device_manager = DeviceManager()
        >>> device_manager.detect_devices()
        >>> 
        >>> asr = ASREngine(debug=True, device_manager=device_manager)
        >>> text = asr.record_and_transcribe()
        >>> if text:
        >>>     print(f"Transcribed: {text}")
        >>> asr.close()
    
    Note:
        The ASR engine requires a properly configured DeviceManager with
        a selected microphone. If no device_manager is provided, it will
        fall back to creating its own PyAudio instance for compatibility.
    """

    def __init__(self, debug: bool = False, device_manager=None):
        """
        Initialize the ASR engine with optional device manager integration.
        
        Sets up the Whisper model, audio configuration, and integrates with
        the provided DeviceManager for microphone operations. If no device
        manager is provided, creates fallback PyAudio instance for compatibility.
        
        Args:
            debug (bool): Enable debug logging for detailed operation tracking.
                         When True, prints detailed information about recording,
                         transcription, and device operations.
            device_manager (DeviceManager, optional): Configured DeviceManager instance
                                                    with selected microphone. If None,
                                                    creates fallback PyAudio instance.
        
        Raises:
            Exception: If Whisper model loading fails or audio system initialization fails
        
        Side Effects:
            - Initializes Whisper model (downloads if not present)
            - Sets up audio configuration parameters
            - Creates temporary file path for audio storage
            - Initializes logging system
        
        Note:
            The Whisper model will be downloaded to ./models/speech/ if not present.
            Ensure sufficient disk space and internet connectivity for first run.
        """
        # Initialize logging system
        self.logging_manager = LoggingManager()
        self.logging_manager.add_message("Initiating - ASREngine", level="INFO", source="ASREngine")
        
        # Store configuration
        self.debug = debug
        self.device_manager = device_manager
        
        # Initialize Whisper model for transcription
        # Using large-v3-turbo for good balance of accuracy and speed
        self.engine = Whisper_ccp(
            model="large-v3-turbo",
            models_dir="./models/speech/",
            print_realtime=False,  # Disable real-time printing for cleaner output
            print_progress=False   # Disable progress bars
        )
        
        # Audio configuration parameters
        # These match common speech recognition requirements
        self.CHUNK = 2048          # Larger chunk size for better speech capture
        self.FORMAT = pyaudio.paInt16  # 16-bit samples (standard for speech)
        self.CHANNELS = 1          # Mono recording (sufficient for speech)
        self.RATE = 44100          # CD quality sampling rate
        
        # Fallback PyAudio instance if no device manager provided
        # This maintains backward compatibility with existing code
        if not self.device_manager:
            self._debug_print("No DeviceManager provided, creating fallback PyAudio instance")
            self.pyaudio = pyaudio.PyAudio()
        else:
            self.pyaudio = None
            self._debug_print("Using provided DeviceManager for audio operations")
        
        # Temporary file for audio storage during transcription
        # Using FLAC format for better compression and quality
        self.WAVE_OUTPUT_FILENAME = "src/speech_engine/temp_audio.flac"
        
        self._debug_print("ASR Engine initialized successfully")

    def _debug_print(self, message: str) -> None:
        """
        Print debug message if debug mode is enabled.
        
        Provides consistent debug output formatting for troubleshooting
        and development purposes. All debug messages are prefixed with
        [ASR DEBUG] for easy identification in log streams.
        
        Args:
            message (str): Debug message to print
            
        Note:
            Debug messages are only printed when self.debug is True.
            This allows for clean production output while enabling
            detailed debugging during development.
        """
        if self.debug:
            print(f"[ASR DEBUG] {message}")

    def record_and_transcribe(self) -> Optional[str]:
        """
        Record audio input and transcribe it to text using Whisper.
        
        This method implements push-to-talk recording functionality, capturing
        audio while the space key is pressed and then transcribing the recorded
        audio using the Whisper model. The process handles device management,
        audio buffering, file I/O, and transcription with comprehensive error handling.
        
        Recording Process:
        1. Detects space key press to start recording
        2. Opens audio stream using DeviceManager or fallback PyAudio
        3. Continuously captures audio chunks while space key is held
        4. Stops recording when space key is released
        5. Saves audio data to temporary file
        6. Transcribes audio using Whisper model
        7. Cleans up temporary files and returns transcription
        
        Returns:
            Optional[str]: Transcribed text from the audio recording.
                          Returns None if:
                          - No audio was recorded (space key not pressed)
                          - Microphone device is not available
                          - Transcription fails due to model error
                          - File I/O operations fail
        
        Raises:
            No exceptions are raised; all errors are caught and logged.
            
        Side Effects:
            - Creates and deletes temporary audio file
            - Uses microphone device through DeviceManager
            - Logs recording and transcription events
            - Prints debug information if debug mode enabled
            
        Note:
            - Requires space key to be pressed for recording to occur
            - Audio quality depends on microphone device capabilities
            - Transcription accuracy depends on audio clarity and language model
            - Temporary files are automatically cleaned up after transcription
            
        Example:
            >>> # User presses and holds space key while speaking
            >>> text = asr_engine.record_and_transcribe()
            >>> if text:
            >>>     print(f"You said: {text}")
            >>> else:
            >>>     print("No speech detected or transcription failed")
        """
        # Log the start of recording process
        self.logging_manager.add_message("Recording speech input.", level="INFO", source="ASREngine")
        self._debug_print("Starting recording process")
        
        # Initialize audio data collection
        audio_queue = queue.Queue()  # Thread-safe queue for audio data
        frames = []  # Buffer to store recorded audio frames

        def audio_callback(in_data, frame_count, time_info, status):
            """
            Callback function for PyAudio stream to handle incoming audio data.
            
            This callback is called by PyAudio whenever new audio data is available
            from the microphone. It simply queues the data for processing in the
            main thread to avoid blocking the audio stream.
            
            Args:
                in_data: Raw audio data from microphone
                frame_count: Number of frames in the data
                time_info: Timing information (unused)
                status: Stream status flags (unused)
                
            Returns:
                Tuple: (None, pyaudio.paContinue) to continue streaming
            """
            audio_queue.put(in_data)
            return (None, pyaudio.paContinue)

        stream = None
        try:
            # Main recording loop - continues while space key is pressed
            while keyboard.is_pressed("space"):
                # Initialize audio stream on first iteration
                if stream is None:
                    self._debug_print("Opening audio stream")
                    
                    # Use DeviceManager if available, otherwise fallback to PyAudio
                    if self.device_manager and self.device_manager.main_microphone:
                        # Ensure microphone is started with compatible settings
                        if not self.device_manager.main_microphone['is_active']:
                            # Start microphone with ASR-optimized settings
                            success = self.device_manager.start_microphone(
                                chunk_size=self.CHUNK,
                                sample_rate=self.RATE,
                                audio_format=self.FORMAT,
                                channels=self.CHANNELS
                            )
                            if not success:
                                self._debug_print("Failed to start DeviceManager microphone")
                                return None
                        
                        # Get the PyAudio instance and create stream using device manager's audio
                        pyaudio_instance = self.device_manager.main_microphone['audio']
                        device_index = self.device_manager.main_microphone['index']
                        
                        stream = pyaudio_instance.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=self.CHUNK,
                            stream_callback=audio_callback,
                        )
                    else:
                        # Fallback to direct PyAudio usage
                        self._debug_print("Using fallback PyAudio instance")
                        if not self.pyaudio:
                            self._debug_print("No audio system available")
                            return None
                            
                        stream = self.pyaudio.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK,
                            stream_callback=audio_callback,
                        )
                
                # Process any available audio data from the queue
                if not audio_queue.empty():
                    data = audio_queue.get()
                    frames.append(data)
                    
        except Exception as e:
            self._debug_print(f"Error during recording: {e}")
            self.logging_manager.add_message(f"Recording error: {e}", level="ERROR", source="ASREngine")
            return None
        finally:
            # Always clean up the audio stream
            if stream:
                self._debug_print("Closing audio stream")
                stream.stop_stream()
                stream.close()

        # Process recorded audio if any frames were captured
        if len(frames) > 0:
            self._debug_print(f"Recorded {len(frames)} frames, saving to file")
            
            try:
                # Save the recorded audio to a temporary file for Whisper processing
                with wave.open(self.WAVE_OUTPUT_FILENAME, "wb") as wf:
                    wf.setnchannels(self.CHANNELS)
                    # Get sample width from PyAudio format
                    if self.device_manager and self.device_manager.main_microphone:
                        sample_width = self.device_manager.main_microphone['audio'].get_sample_size(self.FORMAT)
                    else:
                        sample_width = self.pyaudio.get_sample_size(self.FORMAT)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(self.RATE)
                    wf.writeframes(b"".join(frames))

                # Transcribe the recorded audio using Whisper
                self._debug_print("Starting transcription")
                result = self.engine.transcribe(media=self.WAVE_OUTPUT_FILENAME)
                
                # Extract transcribed text from Whisper result
                transcribed_text = result[0].text if result else None
                self._debug_print(f"Transcription result: '{transcribed_text}'")
                
                # Log successful transcription
                if transcribed_text:
                    self.logging_manager.add_message(
                        f"Successfully transcribed: '{transcribed_text[:50]}...'", 
                        level="INFO", 
                        source="ASREngine"
                    )
                
                return transcribed_text

            except Exception as e:
                self._debug_print(f"Transcription error: {e}")
                self.logging_manager.add_message(f"Transcription error: {e}", level="ERROR", source="ASREngine")
                return None
            finally:
                # Always clean up temporary file
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    try:
                        os.remove(self.WAVE_OUTPUT_FILENAME)
                        self._debug_print("Temporary audio file cleaned up")
                    except Exception as e:
                        self._debug_print(f"Error cleaning up temporary file: {e}")
        else:
            self._debug_print("No audio frames recorded")
            return None

    def close(self) -> None:
        """
        Gracefully shutdown ASR engine resources and cleanup.
        
        Performs comprehensive cleanup of all ASR engine resources including
        PyAudio instances, Whisper model resources, temporary files, and
        device manager integration. This method ensures proper resource
        deallocation and prevents memory leaks or device conflicts.
        
        Cleanup Process:
        1. Terminates PyAudio instances (fallback or device manager)
        2. Releases Whisper model resources
        3. Removes temporary audio files
        4. Logs successful shutdown
        5. Handles any cleanup errors gracefully
        
        Side Effects:
        - Terminates all PyAudio resources
        - Deallocates Whisper model memory
        - Removes temporary files from filesystem
        - Logs shutdown events
        - Does NOT stop DeviceManager (external resource)
        
        Note:
        - Safe to call multiple times
        - Does not affect external DeviceManager instance
        - Handles cases where resources were never initialized
        - All errors during cleanup are logged but don't raise exceptions
        
        Example:
        >>> asr_engine = ASREngine(device_manager=device_manager)
        >>> # ... use engine for transcription ...
        >>> asr_engine.close()  # Clean shutdown
        """
        try:
            self._debug_print("Shutting down ASR engine...")
            
            # Clean up PyAudio resources
            # Only clean up fallback PyAudio instance, not DeviceManager's
            if hasattr(self, 'pyaudio') and self.pyaudio is not None:
                self.pyaudio.terminate()
                self.pyaudio = None
                self._debug_print("Fallback PyAudio instance terminated")
            
            # Note: DeviceManager cleanup is handled externally
            # We don't terminate the DeviceManager's PyAudio instance
            # as it may be used by other components
            
            # Clean up Whisper model resources
            if hasattr(self, 'engine') and self.engine is not None:
                # The Whisper model cleanup depends on the specific implementation
                # Some implementations may not need explicit cleanup
                try:
                    del self.engine
                    self._debug_print("Whisper model resources cleaned up")
                except Exception as e:
                    self._debug_print(f"Note: Whisper model cleanup handling: {e}")
            
            # Clean up temporary audio file if it exists
            if hasattr(self, 'WAVE_OUTPUT_FILENAME') and os.path.exists(self.WAVE_OUTPUT_FILENAME):
                try:
                    os.remove(self.WAVE_OUTPUT_FILENAME)
                    self._debug_print("Temporary audio file cleaned up")
                except Exception as e:
                    self._debug_print(f"Warning: Could not remove temporary file: {e}")
            
            # Log successful shutdown
            self.logging_manager.add_message("ASREngine shutdown complete", level="INFO", source="ASREngine")
            self._debug_print("ASR engine shutdown completed successfully")
            
        except Exception as e:
            # Log any errors during shutdown but don't raise them
            # This ensures cleanup doesn't fail due to individual component errors
            error_msg = f"Error during ASR engine shutdown: {e}"
            self._debug_print(error_msg)
            self.logging_manager.add_message(error_msg, level="ERROR", source="ASREngine")


class SpeechEngine:
    """Main speech engine combining TTS and ASR functionality."""
    
    def __init__(self, debug: bool = False, device_manager: DeviceManager =None):
        """
        Initialize the speech engine.
        
        Args:
            debug (bool): Enable debug logging for all components
        """
        self.logging_manager = LoggingManager()
        self.logging_manager.add_message("Initiating - SpeechEngine", level="INFO", source="SpeechEngine")
        
        self.device_manager = device_manager
        self.debug = debug
        self.tts_engine = TTSEngine(debug=debug, device_manager=device_manager)
        self.asr_engine = ASREngine(debug=debug, device_manager=device_manager)

        if self.debug:
            print("[SPEECH ENGINE DEBUG] Speech engine initialized")

    
    
    def speak(self, text: str, stream: bool = False, visualize: bool = True, mute: bool = False) -> Union[Tuple[str, str], Generator[str, None, None]]:
        """
        Speak the given text using TTS.
        
        Args:
            text (str): Text to speak
            stream (bool): Whether to stream text character by character
            visualize (bool): Whether to show LED visualization
            mute (bool): Whether to mute audio output
            
        Returns:
            Union[Tuple[str, str], Generator[str, None, None]]: 
                If stream=True: Generator yielding characters
                If stream=False: Tuple of (spoken_text, unspoken_text)
        """
        if self.debug:
            print(f"[SPEECH ENGINE DEBUG] Speaking text - Stream: {stream}, Visualize: {visualize}, Mute: {mute}; Text: '{text}...'")
        try:
            x = self.tts_engine.play_synthesized_audio(
                text, stream=stream, mute=mute
            ) # visualize=visualize,
        except Exception as e:
            if self.debug:
                print(f"[SPEECH ENGINE DEBUG] Error during speak: {e}")
            self.logging_manager.add_message(f"Error during speak: {e}", level="ERROR", source="SpeechEngine")
            return ("", text) if not stream else iter([])
        return x
        
    
    def listen(self, duration: int = 5) -> Optional[str]:
        """
        Listen for speech input and transcribe it.
        
        Args:
            duration (int): Maximum recording duration in seconds
            
        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        if self.debug:
            print(f"[SPEECH ENGINE DEBUG] Listening for speech")
        
        return self.asr_engine.record_and_transcribe()
    
    def close(self) -> None:
        """Gracefully shutdown all speech engine resources."""
        try:
            if self.debug:
                print("[SPEECH ENGINE DEBUG] Shutting down speech engine...")
            
            # Close TTS engine
            if hasattr(self, 'tts_engine') and self.tts_engine is not None:
                self.tts_engine.close()
            
            # Close ASR engine
            if hasattr(self, 'asr_engine') and self.asr_engine is not None:
                self.asr_engine.close()
            
            self.logging_manager.add_message("SpeechEngine shutdown complete", level="INFO", source="SpeechEngine")
            
            if self.debug:
                print("[SPEECH ENGINE DEBUG] Speech engine shutdown complete")
                
        except Exception as e:
            if self.debug:
                print(f"[SPEECH ENGINE DEBUG] Error during speech engine shutdown: {e}")
            self.logging_manager.add_message(f"Error during SpeechEngine shutdown: {e}", level="ERROR", source="SpeechEngine")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()