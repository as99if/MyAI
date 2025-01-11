# import subprocess

# subprocess.call(['osascript', '-e', '"set Volume 0"'])

import osascript
# import siri
# osascript.run("set Volume 10")
# osascript.run('tell app "Spotify" to activate')
import keyboard
import time


# https://coolaj86.com/articles/how-to-control-os-x-system-volume-with-applescript/
# increase volume
# "increase" + "volume"
# osascript.run("set volume output volume ( output volume of (get volume settings) + 20)")

# decrease volume
# if  "decrease" in prompt or "volume" in prompt:
# osascript.run("set volume output volume ( output volume of (get volume settings) - 20)")

# set volume to n%
# "volume" + n + "percent"
# osascript.run('quit app "terminal_command.py"')

def command(prompt):

    if "increase" and "volume" in prompt:
        osascript.run("set volume output volume ( output volume of (get volume settings) + 20)")
        return "Volume increased"
    if "decrease" and "volume" in prompt:
        osascript.run("set volume output volume ( output volume of (get volume settings) - 20)")
        return "Volume decreased"
    # if "volume" and "percent" in prompt:
    # check numerical value in str
    # n = num
    # osascript.run(f"set volume output volume {n}")

    else:
        return None


# https://stackoverflow.com/questions/39759230/interacting-with-siri-via-the-command-line-in-macos
# enable Type to Siri (System Preferences > Accessibility > Sir)

def trigger_siri():
    keyboard.press('command+space')
    time.sleep(0.3)
    keyboard.release('command+space')
    time.sleep(0.2)  # Wait for Siri to load


def siri_command(prompt):
    trigger_siri()
    keyboard.write(prompt)
    keyboard.send('enter')
    return "Siri has done it"


#siri_command()

