
import subprocess


def display_notification(title, subtitle, message):
    sound_name = ""
    # Replace "sound_name" with any valid sound name from ~/Library/Sounds or /System/Library/Sounds
    script = f'display notification "{message}" with title "{title}" subtitle "{subtitle}" sound name "{sound_name}"'
   
    try:
        result = subprocess.run(['/usr/bin/osascript', '-e', script], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.rstrip()
        else:
            raise ChildProcessError(f'AppleScript: {result.stderr.rstrip()}')
    except Exception as e:
        print(f"An error occurred: {e}")
        


def display_notification_with_button(title, subtitle, message, buttons=None, button_actions=None):
    """
    Displays a notification on macOS with optional buttons and associated actions.

    Args:
        title: The title of the notification.
        subtitle: The subtitle of the notification.
        message: The message of the notification.
        buttons: A list of button labels (strings).  If None, no buttons are added.
                 Maximum 3 buttons are supported by macOS notifications.
        button_actions: A list of functions corresponding to the buttons.
                        The length of this list must match the length of the 'buttons' list.
                        Each function will be called when the corresponding button is clicked
                        (via AppleScript).  If None, no actions are performed.

    Returns:
        The label of the button that was clicked, or None if the notification was dismissed
        without clicking a button or if an error occurred.
    """
    sound_name = ""  # Replace with a valid sound name if desired
    script = f'display notification "{message}" with title "{title}" subtitle "{subtitle}" sound name "{sound_name}"'

    if buttons:
        if len(buttons) > 3:
            print("Warning: macOS notifications support a maximum of 3 buttons.  Only the first 3 will be used.")
            buttons = buttons[:3]  # Limit to 3 buttons
        button_list = '", "'.join(buttons)  # Format buttons for AppleScript
        script += f' buttons "{button_list}"'
        script += ' default button 1' # Ensure a default button exists in case of keyboard entry
    else:
        script += ' giving up after 5'

    try:
        result = subprocess.run(['/usr/bin/osascript', '-e', script], capture_output=True, text=True)
        if result.returncode == 0:
            button_pressed = result.stdout.rstrip()

            if buttons and button_actions and button_pressed in buttons:
                # Execute the corresponding action
                button_index = buttons.index(button_pressed)
                button_actions[button_index]()  # Call the function

            return button_pressed if buttons else None #added
        else:
            raise ChildProcessError(f'AppleScript: {result.stderr.rstrip()}')
    except Exception as e:
        print(f"An error occurred: {e}")
        return None # added

