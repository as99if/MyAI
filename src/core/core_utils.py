
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
        
