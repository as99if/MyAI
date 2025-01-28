import pyautogui
import base64
from io import BytesIO

def take_screenshot():
    """
    Captures a screenshot of the entire screen and saves it to a file.
    Returns the file path of the saved screenshot.
    """
    try:
        
    # Take a screenshot
        screenshot = pyautogui.screenshot()

        # Convert the screenshot to bytes
        buffered = BytesIO()
        screenshot.save(buffered, format="JPEG")

        # Encode the bytes to base64
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_image
        #print(f"Base64 encoded screenshot: {base64_image[:50]}...")  # Print first 50 characters

    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return None



