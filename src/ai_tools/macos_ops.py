import subprocess
import re

def run_osascript(script):
    """Run the given AppleScript and return the standard output."""
    try:
        result = subprocess.run(['/usr/bin/osascript', '-e', script], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.rstrip()
        else:
            raise ChildProcessError(f'AppleScript: {result.stderr.rstrip()}')
    except Exception as e:
        print(f"An error occurred: {e}")

def perform_action(command):
    actions = {
        'hello': lambda: run_osascript('display dialog "Hello, World!" with title "macOS Operation"'),
        'open terminal': lambda: run_osascript('tell application "Terminal" to activate'),
        'new tab': lambda: run_osascript('tell application "System Events" to tell process "Terminal" to keystroke "t" using command down'),
        'open directory': lambda: run_osascript('tell application "Finder" to open folder "/path/to/directory"'),
        'show desktop': lambda: run_osascript('tell application "System Events" to keystroke "d" using command down and control down'),
        'lock screen': lambda: run_osascript('tell application "System Events" to keystroke "q" using command down and control down'),
        'sleep computer': lambda: run_osascript('tell application "System Events" to sleep'),
        'restart computer': lambda: run_osascript('tell application "System Events" to restart'),
        'shut down computer': lambda: run_osascript('tell application "System Events" to shut down'),
        'log out': lambda: run_osascript('tell application "System Events" to log out'),
        'open url': lambda: run_osascript('open location "https://example.com"'),
        'create new email': lambda: run_osascript('tell application "Mail" to activate') or run_osascript('tell application "System Events" to tell process "Mail" to click menu item "New Message" of menu "File" of menu bar 1'),
        'send notification': lambda: run_osascript('display notification "Hello, World!" with title "Notification"'),
        'get current date': lambda: run_osascript('current date'),
        'set system time': lambda: run_osascript('set theDate to (current date) + 3600') or run_osascript('set system time to theDate'),
        'get running applications': lambda: run_osascript('tell application "System Events" to get name of every process'),
        'hide application': lambda: run_osascript('tell application "TextEdit" to hide'),
        'show hidden files': lambda: run_osascript('tell application "Finder" to set AppleShowAllFiles to true'),
        'hide hidden files': lambda: run_osascript('tell application "Finder" to set AppleShowAllFiles to false'),
        'empty trash': lambda: run_osascript('tell application "Finder" to empty trash'),
        'create new folder': lambda: run_osascript('tell application "Finder" to make new folder at desktop'),
        'rename file': lambda: run_osascript('tell application "Finder" to set name of file "old_name.txt" of desktop to "new_name.txt"'),
        'move file': lambda: run_osascript('tell application "Finder" to move file "file.txt" of desktop to folder "Documents" of desktop'),
        'copy file': lambda: run_osascript('tell application "Finder" to duplicate file "file.txt" of desktop to folder "Documents" of desktop'),
        'delete file': lambda: run_osascript('tell application "Finder" to delete file "file.txt" of desktop'),
        'create alias': lambda: run_osascript('tell application "Finder" to make alias file to file "file.txt" of desktop at desktop'),
        'get file info': lambda: run_osascript('tell application "Finder" to get info for file "file.txt" of desktop'),
        'set file permissions': lambda: run_osascript('do shell script "chmod 755 /path/to/file.txt"'),
        'get list of files': lambda: run_osascript('tell application "Finder" to get name of every file of folder "Documents" of desktop'),
        'get current user': lambda: run_osascript('do shell script "whoami"'),
        'get computer name': lambda: run_osascript('do shell script "scutil --get ComputerName"'),
        'get ip address': lambda: run_osascript('do shell script "ipconfig getifaddr en0"'),
        'set screen saver': lambda: run_osascript('tell application "System Events" to keystroke "s" using command down and control down'),
        'turn on bluetooth': lambda: run_osascript('do shell script "blueutil power 1"'),
        'turn off bluetooth': lambda: run_osascript('do shell script "blueutil power 0"'),
        'turn on wifi': lambda: run_osascript('do shell script "networksetup -setairportpower en0 on"'),
        'turn off wifi': lambda: run_osascript('do shell script "networksetup -setairportpower en0 off"'),
        'get battery level': lambda: run_osascript('do shell script "pmset -g batt | grep -o \".*%\""'),
        'get battery status': lambda: run_osascript('do shell script "pmset -g batt | grep -o \"AC Power\""'),
        'set brightness': lambda: run_osascript('tell application "System Events" to set the brightness of displays to 0.5'),
        'get brightness': lambda: run_osascript('tell application "System Events" to get the brightness of displays'),
        'take screenshot': lambda: run_osascript('do shell script "screencapture ~/Desktop/screenshot.png"'),
    }

    # Check if the command matches any action
    for key in actions:
        if re.search(key, command, re.IGNORECASE):
            print(f"Performing action: {key}")
            actions[key]()
            return

    print("Unknown command.")