import cv2
import numpy as np
import time
from typing import Optional, Tuple
from src.interface.peripheral_manager.devie_management import DeviceManager


class VisionProcessor:
    """
    Vision processor that handles camera feeds and frame processing using DeviceManager.
    # Create and use VisionProcessor
    processor = VisionProcessor()

    # Process video stream with display
    processor.process_video_stream(display=True)

    # Or capture single frame
    frame = processor.capture_single_frame("snapshot.jpg")

    # Cleanup when done
    processor.cleanup()
    """
    
    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """
        Initialize VisionProcessor with a DeviceManager instance.
        
        Args:
            device_manager: Optional DeviceManager instance. If None, creates a new one.
        """
        if device_manager is None:
            self.device_manager = DeviceManager()
            self.device_manager.detect_devices()
        else:
            self.device_manager = device_manager
        
        self.is_processing = False
        self.frame_count = 0
        
    def start_camera_processing(self) -> bool:
        """
        Start camera processing by initializing the camera through DeviceManager.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        if not self.device_manager.main_camera:
            print("No camera selected. Please run detect_devices() first.")
            return False
        
        success = self.device_manager.start_camera()
        if success:
            self.is_processing = True
            print(f"Started processing with camera: {self.device_manager.main_camera['name']}")
        else:
            print("Failed to start camera processing.")
        
        return success
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get a frame from the camera through DeviceManager.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame data
        """
        if not self.is_processing:
            return False, None
        
        ret, frame = self.device_manager.get_frame()
        if ret:
            self.frame_count += 1
        
        return ret, frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame with AI/computer vision algorithms.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            np.ndarray: Processed frame
        """
        if frame is None:
            return frame
        
        # TODO: Add AI processing here (object detection, face recognition, etc.)
        processed_frame = frame.copy()
        
        # Add processing indicators
        height, width = processed_frame.shape[:2]
        
        # Add camera info
        camera_info = f"Camera: {self.device_manager.main_camera['name']}" if self.device_manager.main_camera else "No Camera"
        cv2.putText(processed_frame, camera_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add frame counter
        cv2.putText(processed_frame, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return processed_frame
    
    def process_video_stream(self, display: bool = True, save_video: bool = False, output_path: str = "output.avi"):
        """
        Process video stream continuously.
        
        Args:
            display: Whether to display the processed frames
            save_video: Whether to save processed video to file
            output_path: Path for saved video file
        """
        if not self.start_camera_processing():
            return
        
        # Setup video writer if saving
        video_writer = None
        if save_video:
            # Get camera properties for video writer
            props = self.device_manager.get_camera_properties()
            if props:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, 
                                             (props['width'], props['height']))
        
        try:
            print("Starting video stream processing. Press 'q' to quit.")
            
            while self.is_processing:
                ret, frame = self.get_frame()
                
                if not ret or frame is None:
                    print("Failed to get frame from camera.")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Save frame if requested
                if save_video and video_writer:
                    video_writer.write(processed_frame)
                
                # Display frame if requested
                if display:
                    cv2.imshow('Vision Processor', processed_frame)
                    
                    # Check for quit command
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        
        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
            
            if display:
                cv2.destroyAllWindows()
            
            self.stop_processing()
    
    def capture_single_frame(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Capture and process a single frame.
        
        Args:
            save_path: Optional path to save the processed frame
            
        Returns:
            Optional[np.ndarray]: Processed frame or None if failed
        """
        if not self.is_processing:
            if not self.start_camera_processing():
                return None
        
        ret, frame = self.get_frame()
        if not ret or frame is None:
            print("Failed to capture frame.")
            return None
        
        processed_frame = self.process_frame(frame)
        
        if save_path:
            cv2.imwrite(save_path, processed_frame)
            print(f"Frame saved to: {save_path}")
        
        return processed_frame
    
    def get_camera_info(self) -> dict:
        """
        Get information about the current camera.
        
        Returns:
            dict: Camera information and properties
        """
        if not self.device_manager.main_camera:
            return {}
        
        camera_info = {
            'name': self.device_manager.main_camera['name'],
            'index': self.device_manager.main_camera['index'],
            'is_active': self.device_manager.main_camera.get('is_active', False),
            'frame_count': self.frame_count
        }
        
        # Add camera properties if camera is active
        if self.is_processing:
            camera_info.update(self.device_manager.get_camera_properties())
        
        return camera_info
    
    def stop_processing(self):
        """Stop camera processing and cleanup resources."""
        self.is_processing = False
        self.device_manager.stop_camera()
        print("Vision processor stopped.")
    
    def cleanup(self):
        """Cleanup all resources including DeviceManager."""
        self.stop_processing()
        self.device_manager.cleanup()
        print("Vision processor cleanup completed.")


# Example usage and testing functions
def test_vision_processor():
    """Test function to demonstrate VisionProcessor usage."""
    processor = VisionProcessor()
    
    # Test single frame capture
    print("Testing single frame capture...")
    frame = processor.capture_single_frame("test_frame.jpg")
    if frame is not None:
        print("Single frame captured successfully!")
    
    # Test video stream (uncomment to test)
    # print("Testing video stream...")
    # processor.process_video_stream(display=True, save_video=False)
    
    processor.cleanup()


if __name__ == "__main__":
    test_vision_processor()