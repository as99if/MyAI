import subprocess

def execute_siri_command(command_text):
    """
    Executes a command via Siri using AppleScript and osascript.
    """
    try:
        # AppleScript to trigger Siri and type the command
        script = f"""
        tell application "System Events" to tell the front menu bar of process "SystemUIServer"
            tell (first menu bar item whose description is "Siri")
                perform action "AXPress"
            end tell
        end tell
        delay 2
        tell application "System Events"
            set textToType to "{command_text}"
            keystroke textToType
            key code 36 -- Press Enter
        end tell
        """
        # Run the AppleScript using osascript
        subprocess.run(["osascript", "-e", script], check=True)
        print(f"Siri executed command: {command_text}")
    except Exception as e:
        print(f"Error executing Siri command: {e}")
