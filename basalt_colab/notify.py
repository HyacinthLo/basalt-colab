"""Best-effort terminal bell and OS desktop notifications (zero third-party dependencies)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _os_notify(title: str, body: str) -> None:
    try:
        if sys.platform == "darwin":
            if not shutil.which("osascript"):
                return
            t = title.replace("\\", "\\\\").replace('"', '\\"')
            b = body.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.Popen(
                ["osascript", "-e", f'display notification "{b}" with title "{t}"'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform.startswith("linux"):
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                return
            exe = shutil.which("notify-send")
            if not exe:
                return
            subprocess.Popen(
                [exe, title, body],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif os.name == "nt":
            ps = shutil.which("powershell") or shutil.which("pwsh")
            if not ps:
                return
            t = title.replace("'", "''")
            b = body.replace("'", "''")
            script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$n = New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon = [System.Drawing.SystemIcons]::Information;"
                f"$n.BalloonTipTitle = '{t}';"
                f"$n.BalloonTipText = '{b}';"
                "$n.Visible = $true; $n.ShowBalloonTip(5000);"
            )
            subprocess.Popen(
                [ps, "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except OSError:
        pass


def notify(title: str, body: str, *, enabled: bool = True, bell: bool = True) -> None:
    """Emits a terminal bell (on TTY) and a best-effort non-blocking OS desktop notification."""
    if not enabled or os.environ.get("BASALT_COLAB_NO_NOTIFY") in ("1", "true", "yes"):
        return
    if bell and sys.stdout and sys.stdout.isatty():
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except OSError:
            pass
    _os_notify(title, body)
