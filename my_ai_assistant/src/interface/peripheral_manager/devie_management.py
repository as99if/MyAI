"""
Device Management Module

This module provides comprehensive management of peripheral devices (cameras and microphones)
for AI assistant applications. It handles device detection, selection, initialization, and
data streaming operations.

Features:
- Automatic detection of available cameras and microphones
- Smart selection prioritizing external devices over built-in ones
- Real-time video frame capture and audio data streaming
- Proper resource management and cleanup
- Cross-platform compatibility with Linux systems

Author: Asif Ahmed
Version: 
"""

import cv2
import pyaudio
import subprocess
import numpy as np
from typing import List, Dict, Optional, Tuple

import threading
import queue
import time
try:
    import pynput
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Key input detection will be limited.")


class DeviceManager:
    """
    Unified manager for camera and microphone devices detection, selection, and operation.
    
    This class provides a comprehensive interface for managing peripheral devices
    commonly used in AI applications such as webcams and microphones.
    It automatically detects available devices, prioritizes external devices,
    and provides methods for data capture and streaming.
    
    The class integrates both camera and microphone functionality in a single
    manager, eliminating the need for separate interface classes and providing
    a consistent API for all device operations.
    
    Attributes:
        available_cameras (List[Dict]): List of detected camera devices with metadata
        available_microphones (List[Dict]): List of detected microphone devices with metadata
        main_camera (Optional[Dict]): Currently selected camera device information and state
        main_microphone (Optional[Dict]): Currently selected microphone device information and state
    
    Example:
        >>> device_manager = DeviceManager()
        >>> device_manager.detect_devices()
        >>> device_manager.start_camera()
        >>> device_manager.start_microphone()
        >>> ret, frame = device_manager.get_frame()
        >>> audio_data = device_manager.get_audio()
        >>> device_manager.cleanup()
    """
    
    def __init__(self):
        """
        Initialize the DeviceManager.
        
        Sets up empty lists for devices and initializes main device references to None.
        No device detection is performed during initialization - call detect_devices() explicitly.
        
        The microphone configuration is set with defaults suitable for AI applications:
        - Format: 16-bit PCM (paInt16)
        - Channels: Mono (1 channel)
        - Sample Rate: 44.1 kHz (CD quality)
        - Chunk Size: 1024 frames (good balance of latency and stability)
        """
        # Store all detected devices with their metadata
        self.available_cameras: List[Dict] = []
        self.available_microphones: List[Dict] = []
        self.available_audio_outputs: List[Dict] = []  # New
        self.available_key_devices: List[Dict] = []    # New
        
        # Currently selected devices for active use
        self.main_camera = None
        self.main_microphone = None
        self.main_audio_output = None    # New
        self.main_key_device = None      # New
        
        # Default audio parameters for microphone operations
        # These can be overridden in start_microphone() if needed
        self._default_audio_format = pyaudio.paInt16  # 16-bit samples
        self._default_channels = 1  # Mono recording
        self._default_sample_rate = 44100  # CD quality sampling rate
        self._default_chunk_size = 1024  # Buffer size for audio chunks

        self._recording_button = "space"  # Default button for push-to-record functionality
        self._interrupt_button = ["esc", "space"]  # Default button to interrupt operations

    def detect_cameras(self) -> List[Dict]:
        """
        Detect all available cameras using OpenCV.
        
        This method scans through video device indices (0-9) and attempts to open
        each one using OpenCV's VideoCapture. For each successfully opened device,
        it tries to retrieve the device name using v4l2-ctl (Linux video4linux2 utility).
        
        The method categorizes cameras as 'built-in' (index 0) or 'external' (index > 0),
        which is a heuristic that works well for most laptop configurations.
        
        Returns:
            List[Dict]: List of camera dictionaries containing:
                - index (int): OpenCV device index
                - name (str): Human-readable device name
                - type (str): 'built-in' or 'external'
        
        Note:
            Requires v4l2-utils package on Linux for proper device naming.
            Falls back to generic naming if v4l2-ctl is not available.
        """
        cameras = []
        
        # Scan through common video device indices
        # Most systems don't have more than 10 video devices
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get a descriptive name using Linux video4linux2 utilities
                camera_name = self._get_camera_name_v4l2(i)
                if not camera_name:
                    # Fallback to generic naming if v4l2-ctl fails
                    camera_name = f"Camera {i}"
                
                cameras.append({
                    'index': i,
                    'name': camera_name,
                    # Heuristic: index 0 is usually built-in, others are external
                    'type': 'external' if i > 0 else 'built-in'
                })
                cap.release()  # Important: release the device immediately
        
        # Store detected cameras and return the list
        self.available_cameras = cameras
        return cameras
    
    def _get_camera_name_v4l2(self, index: int) -> Optional[str]:
        """
        Get camera name using v4l2-ctl command (Linux-specific).
        
        This private method uses the v4l2-ctl utility to query device information
        from the Linux video4linux2 subsystem. This provides more descriptive
        device names than generic OpenCV naming.
        
        Args:
            index (int): Video device index (corresponds to /dev/video{index})
        
        Returns:
            Optional[str]: Camera name if successfully retrieved, None otherwise
        
        Note:
            This method gracefully handles cases where v4l2-ctl is not installed
            or the device doesn't support the query.
        """
        try:
            # Construct the device path (Linux video4linux2 convention)
            device_path = f"/dev/video{index}"
            
            # Execute v4l2-ctl command to get device information
            result = subprocess.run(['v4l2-ctl', '--device', device_path, '--info'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse the output to extract the card type (device name)
                for line in result.stdout.split('\n'):
                    if 'Card type' in line:
                        # Extract everything after the colon and strip whitespace
                        return line.split(':')[1].strip()
        except Exception:
            # Silently fail if v4l2-ctl is not available or other errors occur
            pass
        return None

    def detect_microphones(self) -> List[Dict]:
        """
        Detect all available microphones using PyAudio.
        
        This method enumerates all audio devices available to PyAudio and filters
        for input devices (those with input channels > 0). It categorizes devices
        as 'external' or 'built-in' based on naming patterns commonly used by
        USB and webcam audio devices.
        
        Returns:
            List[Dict]: List of microphone dictionaries containing:
                - index (int): PyAudio device index
                - name (str): Device name from audio system
                - type (str): 'built-in' or 'external'
        
        Note:
            The categorization heuristic looks for 'USB' or 'webcam' in device names
            to identify external devices. This works well for most common hardware.
        """
        microphones = []
        # Initialize PyAudio to query available devices
        audio = pyaudio.PyAudio()
        
        try:
            # Enumerate all audio devices
            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                
                # Only consider devices that can record audio (have input channels)
                if device_info['maxInputChannels'] > 0:  # Input device
                    microphones.append({
                        'index': i,
                        'name': device_info['name'],
                        # Heuristic: USB devices and webcams are usually external
                        'type': 'external' if 'USB' in device_info['name'].upper() or 'webcam' in device_info['name'].lower() else 'built-in'
                    })
        finally:
            # Always clean up PyAudio resources
            audio.terminate()
        
        # Store detected microphones and return the list
        self.available_microphones = microphones
        return microphones
    
    def select_external_camera(self) -> bool:
        """
        Select the best available camera, preferring external devices.
        
        This method implements a smart selection strategy:
        1. First, try to select an external camera (USB webcam)
        2. If no external camera is found, fall back to built-in camera
        3. If no cameras are found at all, report failure
        
        The selected camera is stored in main_camera with its metadata and
        state information needed for operation.
        
        Returns:
            bool: True if a camera was successfully selected, False otherwise
        
        Side Effects:
            - Sets self.main_camera to the selected device
            - Prints selection result to console
        """
        # Filter for external cameras (preferred)
        external_cameras = [cam for cam in self.available_cameras if cam['type'] == 'external']
        
        if external_cameras:
            # Select the first external camera found
            selected_camera = external_cameras[0]
            self.main_camera = {
                'index': selected_camera['index'],
                'name': selected_camera['name'],
                'cap': None,  # OpenCV VideoCapture object (initialized when started)
                'is_active': False  # Whether the camera is currently capturing
            }
            print(f"Selected external camera: {selected_camera['name']}")
            return True
        else:
            # Fallback to any available camera (usually built-in)
            if self.available_cameras:
                fallback_camera = self.available_cameras[0]
                self.main_camera = {
                    'index': fallback_camera['index'],
                    'name': fallback_camera['name'],
                    'cap': None,
                    'is_active': False
                }
                print(f"No external camera found. Using: {fallback_camera['name']}")
                return True
        
        # No cameras available at all
        print("No cameras found!")
        return False
    
    def select_external_microphone(self) -> bool:
        """
        Select the best available microphone, preferring external devices.
        
        Similar to camera selection, this method implements a smart strategy:
        1. First, try to select an external microphone (USB or webcam mic)
        2. If no external microphone is found, fall back to built-in microphone
        3. If no microphones are found at all, report failure
        
        The selected microphone is stored in main_microphone with its metadata,
        state information, and audio configuration needed for recording operations.
        
        Returns:
            bool: True if a microphone was successfully selected, False otherwise
        
        Side Effects:
            - Sets self.main_microphone to the selected device
            - Prints selection result to console
        """
        # Filter for external microphones (preferred)
        external_mics = [mic for mic in self.available_microphones if mic['type'] == 'external']
        
        if external_mics:
            # Select the first external microphone found
            selected_mic = external_mics[0]
            self.main_microphone = {
                'index': selected_mic['index'],
                'name': selected_mic['name'],
                'audio': None,  # PyAudio instance (initialized when started)
                'stream': None,  # PyAudio stream object for recording
                'is_active': False,  # Whether the microphone is currently recording
                # Audio configuration (can be overridden in start_microphone)
                'format': self._default_audio_format,
                'channels': self._default_channels,
                'sample_rate': self._default_sample_rate,
                'chunk_size': self._default_chunk_size,
                'audio_data': []  # Buffer for collected audio data if needed
            }
            print(f"Selected external microphone: {selected_mic['name']}")
            return True
        else:
            # Fallback to any available microphone (usually built-in)
            if self.available_microphones:
                fallback_mic = self.available_microphones[0]
                self.main_microphone = {
                    'index': fallback_mic['index'],
                    'name': fallback_mic['name'],
                    'audio': None,
                    'stream': None,
                    'is_active': False,
                    'format': self._default_audio_format,
                    'channels': self._default_channels,
                    'sample_rate': self._default_sample_rate,
                    'chunk_size': self._default_chunk_size,
                    'audio_data': []
                }
                print(f"No external microphone found. Using: {fallback_mic['name']}")
                return True
        
        # No microphones available at all
        print("No microphones found!")
        return False
    
    def start_camera(self) -> bool:
        """
        Initialize and start the main camera for video capture.
        
        This method creates an OpenCV VideoCapture object for the selected camera
        and configures it with optimal settings for AI applications:
        - Resolution: 1280x720 (HD)
        - Frame rate: 30 FPS
        - Standard format optimizations
        
        The camera must be selected (via detect_devices or manual selection)
        before calling this method.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        
        Side Effects:
            - Creates VideoCapture object and stores it in main_camera['cap']
            - Sets main_camera['is_active'] to True on success
            - Prints error messages on failure
        
        Note:
            The camera settings may not be supported by all devices.
            The method will still succeed if the camera opens, even if
            some property settings fail.
        """
        # Ensure a camera has been selected
        if not self.main_camera:
            return False
        
        try:
            # Create OpenCV VideoCapture object
            self.main_camera['cap'] = cv2.VideoCapture(self.main_camera['index'])
            
            # Verify the camera actually opened
            if not self.main_camera['cap'].isOpened():
                return False
            
            # Configure camera properties for optimal quality
            # These settings are commonly supported and provide good balance
            # between quality and performance for AI applications
            self.main_camera['cap'].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.main_camera['cap'].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.main_camera['cap'].set(cv2.CAP_PROP_FPS, 30)
            
            # Mark camera as active
            self.main_camera['is_active'] = True
            return True
            
        except Exception as e:
            print(f"Error starting camera {self.main_camera['index']}: {e}")
            return False
    
    def start_microphone(self, chunk_size: Optional[int] = None, 
                        sample_rate: Optional[int] = None,
                        audio_format: Optional[int] = None,
                        channels: Optional[int] = None) -> bool:
        """
        Initialize and start the main microphone for audio recording.
        
        This method creates a PyAudio stream for the selected microphone with
        configurable audio parameters. The default settings are optimized for
        speech recognition and general audio processing tasks.
        
        Args:
            chunk_size (Optional[int]): Number of frames per buffer
                                      Uses default (1024) if None
            sample_rate (Optional[int]): Audio sampling frequency in Hz
                                       Uses default (44100) if None
            audio_format (Optional[int]): PyAudio format constant
                                        Uses default (paInt16) if None
            channels (Optional[int]): Number of audio channels
                                    Uses default (1 - mono) if None
        
        Returns:
            bool: True if microphone started successfully, False otherwise
        
        Side Effects:
            - Creates PyAudio instance and stream, stores in main_microphone
            - Updates microphone configuration with provided parameters
            - Sets main_microphone['is_active'] to True on success
            - Prints error messages on failure
        
        Note:
            If parameters are provided, they override the defaults and are
            stored in the microphone configuration for consistency.
            
        Example:
            >>> # Use defaults (44.1kHz, 16-bit, mono)
            >>> device_manager.start_microphone()
            >>> 
            >>> # Custom configuration for speech recognition
            >>> device_manager.start_microphone(sample_rate=16000, chunk_size=512)
        """
        # Ensure a microphone has been selected
        if not self.main_microphone:
            return False
        
        try:
            # Update configuration with provided parameters or use defaults
            if chunk_size is not None:
                self.main_microphone['chunk_size'] = chunk_size
            if sample_rate is not None:
                self.main_microphone['sample_rate'] = sample_rate
            if audio_format is not None:
                self.main_microphone['format'] = audio_format
            if channels is not None:
                self.main_microphone['channels'] = channels
            
            # Create PyAudio instance for this microphone
            self.main_microphone['audio'] = pyaudio.PyAudio()
            
            # Open audio stream with configured parameters
            self.main_microphone['stream'] = self.main_microphone['audio'].open(
                format=self.main_microphone['format'],
                channels=self.main_microphone['channels'],
                rate=self.main_microphone['sample_rate'],
                input=True,  # This is an input (recording) stream
                input_device_index=self.main_microphone['index'],
                frames_per_buffer=self.main_microphone['chunk_size']
            )
            
            # Mark microphone as active and clear any previous audio data
            self.main_microphone['is_active'] = True
            self.main_microphone['audio_data'] = []
            return True
            
        except Exception as e:
            print(f"Error starting microphone {self.main_microphone['index']}: {e}")
            return False
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the main camera.
        
        This method reads one frame from the active camera stream. It's the
        primary method for getting video data for processing, display, or
        streaming applications.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: 
                - bool: True if frame was successfully captured, False otherwise
                - np.ndarray: The captured frame as a numpy array (BGR format),
                             or None if capture failed
        
        Note:
            - The camera must be started (start_camera()) before calling this method
            - Frames are returned in OpenCV's default BGR color format
            - For continuous video processing, call this method in a loop
        
        Example:
            >>> ret, frame = device_manager.get_frame()
            >>> if ret:
            >>>     # Process the frame
            >>>     cv2.imshow('Video', frame)
        """
        # Check if camera is properly initialized and active
        if not self.main_camera or not self.main_camera['is_active'] or not self.main_camera['cap']:
            return False, None
        
        # Read frame from camera
        ret, frame = self.main_camera['cap'].read()
        return ret, frame
    
    def get_audio(self, chunk_size: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Read audio data from the main microphone.
        
        This method reads a chunk of audio samples from the active microphone
        stream. The audio data is returned as a numpy array suitable for
        processing by speech recognition, audio analysis, or other AI models.
        
        Args:
            chunk_size (Optional[int]): Number of audio frames to read
                                      Uses microphone's configured chunk_size if None
        
        Returns:
            Optional[np.ndarray]: Audio data as numpy array with dtype matching
                                the microphone's format, or None if reading failed
        
        Note:
            - The microphone must be started before calling this method
            - Audio data format matches the microphone's configuration
            - For continuous audio processing, call this method in a loop
            - The method includes overflow protection for robust operation
        
        Example:
            >>> audio_data = device_manager.get_audio()
            >>> if audio_data is not None:
            >>>     # Process audio data (e.g., speech recognition)
            >>>     process_audio(audio_data)
        """
        # Check if microphone is properly initialized and active
        if not self.main_microphone or not self.main_microphone['is_active'] or not self.main_microphone['stream']:
            return None
        
        try:
            # Use provided chunk_size or fall back to configured value
            read_chunk_size = chunk_size or self.main_microphone['chunk_size']
            
            # Read audio data from the stream
            # exception_on_overflow=False prevents crashes from audio buffer overruns
            audio_data = self.main_microphone['stream'].read(read_chunk_size, exception_on_overflow=False)
            
            # Convert raw audio bytes to numpy array
            # The dtype should match the audio format (int16 for paInt16)
            if self.main_microphone['format'] == pyaudio.paInt16:
                return np.frombuffer(audio_data, dtype=np.int16)
            elif self.main_microphone['format'] == pyaudio.paFloat32:
                return np.frombuffer(audio_data, dtype=np.float32)
            else:
                # Default to int16 for other formats
                return np.frombuffer(audio_data, dtype=np.int16)
            
        except Exception as e:
            print(f"Error reading audio data: {e}")
            return None
    
    def start_recording(self) -> bool:
        """
        Legacy method name for compatibility. Calls start_microphone().
        
        This method provides backward compatibility for code that might
        expect the MicrophoneInterface naming convention.
        
        Returns:
            bool: True if microphone started successfully, False otherwise
        
        Note:
            This is equivalent to calling start_microphone() with default parameters.
        """
        return self.start_microphone()
    
    def read_audio(self) -> Optional[np.ndarray]:
        """
        Legacy method name for compatibility. Calls get_audio().
        
        This method provides backward compatibility for code that might
        expect the MicrophoneInterface naming convention.
        
        Returns:
            Optional[np.ndarray]: Audio data or None if reading failed
        
        Note:
            This is equivalent to calling get_audio() with default chunk size.
        """
        return self.get_audio()
    
    def stop_camera(self):
        """
        Stop the camera capture and release resources.
        
        This method safely stops the camera stream and releases the OpenCV
        VideoCapture object. It's important to call this method when done
        with the camera to free system resources and allow other applications
        to access the device.
        
        Side Effects:
            - Sets main_camera['is_active'] to False
            - Releases the VideoCapture object
            - Sets main_camera['cap'] to None
        
        Note:
            This method is safe to call multiple times or when the camera
            is not active. It will gracefully handle all states.
        """
        if self.main_camera and self.main_camera['cap']:
            # Mark camera as inactive first
            self.main_camera['is_active'] = False
            
            # Release the OpenCV VideoCapture object
            self.main_camera['cap'].release()
            
            # Clear the reference
            self.main_camera['cap'] = None
    
    def stop_microphone(self):
        """
        Stop the microphone recording and release resources.
        
        This method safely stops the audio stream and releases all PyAudio
        resources. It's important to call this method when done with the
        microphone to prevent audio system issues and resource leaks.
        
        The method performs cleanup in the proper order:
        1. Mark microphone as inactive
        2. Stop and close the audio stream
        3. Terminate the PyAudio instance
        
        Side Effects:
            - Sets main_microphone['is_active'] to False
            - Stops, closes, and clears the audio stream
            - Terminates and clears the PyAudio instance
            - Preserves audio_data buffer for potential later access
        
        Note:
            This method is safe to call multiple times or when the microphone
            is not active. It will gracefully handle all states.
        """
        if self.main_microphone:
            # Mark microphone as inactive first
            self.main_microphone['is_active'] = False
            
            # Stop and close the audio stream if it exists
            if self.main_microphone['stream']:
                self.main_microphone['stream'].stop_stream()
                self.main_microphone['stream'].close()
                self.main_microphone['stream'] = None
            
            # Terminate the PyAudio instance if it exists
            if self.main_microphone['audio']:
                self.main_microphone['audio'].terminate()
                self.main_microphone['audio'] = None
    
    def stop_recording(self):
        """
        Legacy method name for compatibility. Calls stop_microphone().
        
        This method provides backward compatibility for code that might
        expect the MicrophoneInterface naming convention.
        
        Note:
            This is equivalent to calling stop_microphone().
        """
        self.stop_microphone()
    
    def get_camera_properties(self) -> Dict:
        """
        Get current camera properties and settings.
        
        This method retrieves various properties from the active camera,
        providing information about the current video stream configuration.
        This is useful for debugging, logging, or adapting processing
        algorithms to the actual camera capabilities.
        
        Returns:
            Dict: Dictionary containing camera properties:
                - width (int): Frame width in pixels
                - height (int): Frame height in pixels  
                - fps (int): Frames per second
                - brightness (float): Brightness setting
                - contrast (float): Contrast setting
                
                Returns empty dict if camera is not active.
        
        Note:
            Not all properties may be supported by all cameras.
            Some values might return -1 or default values if the
            camera doesn't support querying that property.
        
        Example:
            >>> props = device_manager.get_camera_properties()
            >>> print(f"Camera resolution: {props['width']}x{props['height']}")
        """
        # Check if camera is active and available
        if not self.main_camera or not self.main_camera['cap']:
            return {}
        
        # Get reference to the VideoCapture object
        cap = self.main_camera['cap']
        
        # Query various camera properties
        return {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
        }
    
    def get_microphone_properties(self) -> Dict:
        """
        Get current microphone properties and configuration.
        
        This method returns the current audio configuration for the selected
        microphone, including format, sampling rate, and other parameters.
        
        Returns:
            Dict: Dictionary containing microphone properties:
                - index (int): PyAudio device index
                - name (str): Device name
                - format (int): PyAudio format constant
                - channels (int): Number of audio channels
                - sample_rate (int): Sampling frequency in Hz
                - chunk_size (int): Buffer size in frames
                - is_active (bool): Whether microphone is currently recording
                
                Returns empty dict if no microphone is selected.
        
        Example:
            >>> props = device_manager.get_microphone_properties()
            >>> print(f"Mic: {props['name']} @ {props['sample_rate']}Hz")
        """
        if not self.main_microphone:
            return {}
        
        return {
            'index': self.main_microphone['index'],
            'name': self.main_microphone['name'],
            'format': self.main_microphone['format'],
            'channels': self.main_microphone['channels'],
            'sample_rate': self.main_microphone['sample_rate'],
            'chunk_size': self.main_microphone['chunk_size'],
            'is_active': self.main_microphone['is_active']
        }



    def detect_audio_outputs(self) -> List[Dict]:
        """
        Detect all available audio output devices using PyAudio.
        
        Returns:
            List[Dict]: List of audio output devices with index, name, and type
        """
        audio_outputs = []
        audio = pyaudio.PyAudio()
        
        try:
            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                
                if device_info['maxOutputChannels'] > 0:  # Output device
                    device_name = device_info['name']
                    # Internal devices are preferred
                    is_internal = not any(keyword in device_name.upper() for keyword in 
                                        ['USB', 'HDMI', 'BLUETOOTH', 'EXTERNAL'])
                    
                    audio_outputs.append({
                        'index': i,
                        'name': device_name,
                        'type': 'internal' if is_internal else 'external'
                    })
        finally:
            audio.terminate()
        
        self.available_audio_outputs = audio_outputs
        return audio_outputs

    def detect_key_devices(self) -> List[Dict]:
        """
        Detect available keyboard and key input devices.
        
        Returns:
            List[Dict]: List of key input devices with index, name, and type
        """
        key_devices = []
        
        # Try Linux-specific detection first
        try:
            with open('/proc/bus/input/devices', 'r') as f:
                content = f.read()
            
            device_blocks = content.strip().split('\n\n')
            
            for block in device_blocks:
                device_info = {}
                lines = block.split('\n')
                
                for line in lines:
                    if line.startswith('N: Name='):
                        device_info['name'] = line.split('Name=')[1].strip(' "')
                    elif line.startswith('H: Handlers='):
                        handlers = line.split('Handlers=')[1]
                        if 'kbd' in handlers or any(f'event{i}' in handlers for i in range(20)):
                            device_info['has_keyboard'] = True
                
                if device_info.get('has_keyboard') and device_info.get('name'):
                    device_name = device_info['name']
                    
                    # Categorize device type
                    device_type = 'keyboard'
                    if any(keyword in device_name.lower() for keyword in ['macro', 'pad', 'streamdeck']):
                        device_type = 'macro_pad'
                    elif any(keyword in device_name.lower() for keyword in ['button', 'control']):
                        device_type = 'button_pad'
                    
                    # External devices are preferred
                    is_external = any(keyword in device_name.lower() for keyword in 
                                    ['usb', 'wireless', 'bluetooth', 'external'])
                    
                    key_devices.append({
                        'index': len(key_devices),
                        'name': device_name,
                        'type': device_type,
                        'category': 'external' if is_external else 'internal'
                    })
        
        except Exception:
            # Fallback: Add generic keyboard if pynput is available
            if PYNPUT_AVAILABLE:
                key_devices.append({
                    'index': 0,
                    'name': 'Generic Keyboard',
                    'type': 'keyboard',
                    'category': 'internal'
                })
        
        self.available_key_devices = key_devices
        return key_devices

    def select_audio_output(self) -> bool:
        """
        Select the best audio output device, prioritizing internal devices.
        
        Returns:
            bool: True if an audio output was selected, False otherwise
        """
        if not hasattr(self, 'available_audio_outputs') or not self.available_audio_outputs:
            return False
        
        # Prioritize internal devices
        internal_outputs = [out for out in self.available_audio_outputs if out['type'] == 'internal']
        
        if internal_outputs:
            selected_output = internal_outputs[0]
        else:
            selected_output = self.available_audio_outputs[0]
        
        self.main_audio_output = {
            'index': selected_output['index'],
            'name': selected_output['name'],
            'type': selected_output['type'],
            'is_active': False
        }
        
        print(f"Selected audio output: {selected_output['name']}")
        return True

    def select_key_device(self) -> bool:
        """
        Select the best key input device, prioritizing external devices.
        
        Returns:
            bool: True if a key device was selected, False otherwise
        """
        if not hasattr(self, 'available_key_devices') or not self.available_key_devices:
            return False
        
        # Prioritize external devices
        external_devices = [dev for dev in self.available_key_devices if dev['category'] == 'external']
        
        if external_devices:
            selected_device = external_devices[0]
        else:
            selected_device = self.available_key_devices[0]
        
        self.main_key_device = {
            'index': selected_device['index'],
            'name': selected_device['name'],
            'type': selected_device['type'],
            'listener': None,
            'is_active': False,
            'key_queue': queue.Queue()
        }
        
        print(f"Selected key device: {selected_device['name']}")
        return True

    def start_key_monitoring(self) -> bool:
        """
        Start monitoring key input from the selected device.
        
        Returns:
            bool: True if key monitoring started successfully, False otherwise
        """
        if not PYNPUT_AVAILABLE or not hasattr(self, 'main_key_device') or not self.main_key_device:
            return False
        
        try:
            def on_key_press(key):
                self.main_key_device['key_queue'].put({
                    'key': key,
                    'timestamp': time.time()
                })
            
            self.main_key_device['listener'] = keyboard.Listener(on_press=on_key_press)
            self.main_key_device['listener'].start()
            self.main_key_device['is_active'] = True
            
            return True
        except Exception as e:
            print(f"Error starting key monitoring: {e}")
            return False

    def get_pressed_key(self, timeout: float = 0.0):
        """
        Get the next pressed key from the monitoring queue.
        
        Args:
            timeout (float): Time to wait for a key press (0.0 = non-blocking)
        
        Returns:
            Dict or None: Key event information or None if no key available
        """
        if (not hasattr(self, 'main_key_device') or 
            not self.main_key_device or 
            not self.main_key_device['is_active']):
            return None
        
        try:
            if timeout == 0.0:
                return self.main_key_device['key_queue'].get_nowait()
            else:
                return self.main_key_device['key_queue'].get(timeout=timeout)
        except queue.Empty:
            return None

    def test_key_input(self, duration: float = 10.0):
        """
        Test function to monitor and print pressed keys for development.
        
        Args:
            duration (float): How long to monitor for key presses in seconds
        """
        if not hasattr(self, 'main_key_device') or not self.main_key_device:
            print("No key device selected. Run detect_devices() first.")
            return
        
        was_active = self.main_key_device.get('is_active', False)
        
        try:
            if not was_active:
                if not self.start_key_monitoring():
                    print("Failed to start key monitoring.")
                    return
            
            print(f"Testing key input for {duration} seconds...")
            print("Press any keys (Ctrl+C to exit):")
            
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    key_event = self.get_pressed_key(timeout=0.1)
                    
                    if key_event:
                        key = key_event['key']
                        try:
                            if hasattr(key, 'char') and key.char is not None:
                                key_str = f"'{key.char}'"
                            else:
                                key_str = str(key).replace('Key.', '')
                        except AttributeError:
                            key_str = str(key)
                        
                        print(f"Key pressed: {key_str}")
                    
                except KeyboardInterrupt:
                    print("\nTest interrupted.")
                    break
            
            print("Key input test completed.")
            
        finally:
            if not was_active and hasattr(self, 'main_key_device') and self.main_key_device:
                self.stop_key_monitoring()

    def stop_key_monitoring(self):
        """Stop key monitoring and release resources."""
        if hasattr(self, 'main_key_device') and self.main_key_device:
            self.main_key_device['is_active'] = False
            
            if self.main_key_device['listener']:
                self.main_key_device['listener'].stop()
                self.main_key_device['listener'] = None
    
    def detect_devices(self):
        """
        Detect all devices (cameras, microphones, audio outputs, key inputs) and auto-select devices.
        
        This is the main entry point for device detection. It performs a complete
        scan of available devices, displays the results, and automatically selects
        the best available devices.
        
        Side Effects:
            - Updates all available_* device lists
            - Automatically selects optimal devices for each category
            - Prints detection results to console
        """
        # Detect cameras
        print("Detecting cameras...")
        cameras = self.detect_cameras()
        for cam in cameras:
            print(f"Found camera: {cam['name']} (Index: {cam['index']}, Type: {cam['type']})")
        
        # Detect microphones
        print("\nDetecting microphones...")
        mics = self.detect_microphones()
        for mic in mics:
            print(f"Found microphone: {mic['name']} (Index: {mic['index']}, Type: {mic['type']})")
        
        # Detect audio outputs
        print("\nDetecting audio outputs...")
        outputs = self.detect_audio_outputs()
        for output in outputs:
            print(f"Found audio output: {output['name']} (Index: {output['index']}, Type: {output['type']})")
        
        # Detect key devices
        print("\nDetecting key devices...")
        key_devices = self.detect_key_devices()
        for device in key_devices:
            print(f"Found key device: {device['name']} (Type: {device['type']}, Category: {device['category']})")
        
        # Auto-select devices
        print("\nAuto-selecting devices...")
        self.select_external_camera()
        self.select_external_microphone()
        self.select_audio_output()
        self.select_key_device()

    # Update the cleanup method:
    def cleanup(self):
        """
        Clean up all resources and stop all active devices.
        
        This method provides a convenient way to shut down all device operations
        and release all system resources. It should be called when the application
        is shutting down or when the DeviceManager is no longer needed.
        
        The method ensures proper cleanup order and is safe to call at any time,
        regardless of the current device states.
        
        Side Effects:
            - Stops camera capture if active
            - Stops microphone recording if active
            - Stops key monitoring if active
            - Releases all associated system resources
            - Preserves device selection information
        
        Note:
            After calling cleanup(), the devices can be restarted by calling
            the appropriate start_* methods again. The device selection
            information is preserved.
        
        Example:
            >>> device_manager = DeviceManager()
            >>> # ... use devices ...
            >>> device_manager.cleanup()  # Clean shutdown
        """
        # Stop all device operations
        self.stop_camera()
        self.stop_microphone()
        self.stop_key_monitoring()  # New cleanup