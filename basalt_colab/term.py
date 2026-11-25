"""Terminal UI helpers: ANSI styling, cross-platform glyphs, live status line, prompts."""
from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from typing import Any

from basalt_colab.paths import IS_WIN

TTY: list[bool] = [False]
COLOR: list[bool] = [False]


def init_console() -> None:
    TTY[0] = bool(sys.stdout and sys.stdout.isatty())
    if not TTY[0] or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return
    if sys.platform == "win32":
        try:
            import ctypes

            k32, mode = ctypes.windll.kernel32, ctypes.c_uint32()
            h = k32.GetStdHandle(-11)
            if not (k32.GetConsoleMode(h, ctypes.byref(mode)) and k32.SetConsoleMode(h, mode.value | 4)):
                return
        except (AttributeError, OSError):
            return
    COLOR[0] = True


def ring_bell(enabled: bool = True) -> None:
    if not enabled or not TTY[0] or os.environ.get("BASALT_COLAB_NO_NOTIFY") in ("1", "true", "yes"):
        return
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except OSError:
        pass


def paint(text: str, *styles: str) -> str:
    codes = {"bold": "1", "dim": "2", "red": "31", "green": "32", "cyan": "36", "yellow": "33"}
    return f"\x1b[{';'.join(codes[s] for s in styles)}m{text}\x1b[0m" if COLOR[0] else text


def glyph(fancy: str, plain: str) -> str:
    utf = "utf" in (getattr(sys.stdout, "encoding", None) or "").lower()
    return fancy if utf and (not IS_WIN or os.environ.get("WT_SESSION")) else plain


def units(cu: float) -> str:
    return f"about {cu:.2f} compute units" if cu < 1 else f"about {cu:.1f} compute units"


def minutes(n: float) -> str:
    return f"{n:g} minute{'' if n == 1 else 's'}"


def cli_name() -> str:
    if shutil.which("basalt-colab"):
        return "basalt-colab"
    return "python -m basalt_colab" if IS_WIN else "python3 -m basalt_colab"


class Live:
    def __init__(self) -> None:
        self.width, self.n = 0, 0

    def clear(self) -> None:
        if self.width:
            sys.stdout.write("\r" + " " * self.width + "\r")
            self.width = 0

    def put(self, parts: list[tuple[str, tuple[str, ...]]]) -> None:
        if not TTY[0]:
            return
        cols = shutil.get_terminal_size((80, 24)).columns - 1
        plain = "".join(t for t, _ in parts)
        self.clear()
        sys.stdout.write(plain[:cols] if len(plain) > cols else "".join(paint(t, *st) for t, st in parts))
        sys.stdout.flush()
        self.width = min(len(plain), cols)

    def show(self, when: str, text: str) -> None:
        spin = glyph("⣾⣽⣻⢿⡿⣟⣯⣷", "|/-\\")
        self.n += 1
        self.put([("  ", ()), (f"{when:>5}", ("dim",)), ("  ", ()), (spin[self.n % len(spin)], ("cyan",)), ("  " + text, ())])

    def line(self, text: str) -> None:
        self.clear()
        print(text, flush=True)


def confirm(question: str) -> bool:
    if os.environ.get("BASALT_COLAB_YES") == "1" or os.environ.get("LLM_YES") == "1":
        return True
    if not (sys.stdin and sys.stdin.isatty()):
        return False
    try:
        return input(question + " [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def pick(label: str, hint: str, default: Any, parse: Callable[[str], Any]) -> Any:
    d_str = f"[{default}]"
    row = (
        f"  {paint(f'{label:<14}', 'bold')}{paint(f'{hint:<30}', 'dim')}"
        f"{paint(f'{d_str:>10}', 'cyan')} {paint(glyph('›', '>'), 'cyan')} "
    )
    while True:
        try:
            text = input(row).strip()
        except EOFError:
            text = ""
        val = parse(text or str(default))
        if val is not None:
            return val
        print(f"  {' ' * 14}{paint('not valid, try again', 'red')}")
