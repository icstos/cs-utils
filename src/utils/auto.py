"""
Auto - Automation utilities built on pynput + tkinter (Python 3.12+)

Features:
- Mouse: mouse control + event listening (context manager)
- KeyBoard: keyboard control + event listening (context manager)
- Msg: message dialogs (tkinter, pyautogui-like)
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal, override

import ctypes
from pynput import keyboard, mouse
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button as MouseButton

# Windows: align Listener and Controller coordinates (recommended by pynput)
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor V2
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

# ============ Type aliases (PEP 695) ============
type Point = tuple[int, int]
type Region = tuple[int, int, int, int]  # (x, y, width, height)
type KeyName = str | Key | KeyCode
type ResolvedKey = Key | KeyCode | str

# ============ Constants ============
_BUTTON_MAP: dict[str, MouseButton] = {
    "left": MouseButton.left,
    "right": MouseButton.right,
    "middle": MouseButton.middle,
}
_REVERSE_BUTTON_MAP: dict[MouseButton, str] = {v: k for k, v in _BUTTON_MAP.items()}

_SPECIAL_KEYS: dict[str, Key] = {
    name.lower(): getattr(Key, name)
    for name in dir(Key)
    if not name.startswith("_") and isinstance(val := getattr(Key, name), Key)
}
_KEY_ALIASES: dict[str, Key] = {
    "win": Key.cmd,
    "windows": Key.cmd,
    "super": Key.cmd,
    "cmd": Key.cmd,
    "return": Key.enter,
}

# Reuse Controller instances to avoid repeated creation (pynput recommends keeping them long-lived)
_mouse = mouse.Controller()
_kb = keyboard.Controller()

_pause: float = 0.05


def set_pause(seconds: float) -> None:
    """Set the global pause duration after each automation action."""
    global _pause
    _pause = float(seconds)


def get_pause() -> float:
    return _pause


def _sleep(duration: float | None = None) -> None:
    d = duration if duration is not None else _pause
    if d > 0:
        time.sleep(d)


def _resolve_key(key: KeyName) -> ResolvedKey:
    """Resolve key names: supports 'enter', Key.enter, 'win', etc."""
    match key:
        case str():
            low = key.lower().lstrip("<").rstrip(">")
            return _KEY_ALIASES.get(low) or _SPECIAL_KEYS.get(low) or key
        case _:
            return key


def _key_to_str(key: Key | KeyCode | None) -> str:
    match key:
        case None:
            return "unknown"
        case KeyCode():
            return key.char if key.char is not None else str(key)
        case _:
            return str(key).replace("Key.", "")


def _resolve_button(button: str) -> MouseButton:
    return _BUTTON_MAP.get(button.lower(), MouseButton.left)


def _ease_move(
    ctrl: mouse.Controller,
    x: int,
    y: int,
    duration: float,
    *,
    relative: bool = False,
) -> None:
    """Move with ease-out easing; if duration <= 0, move instantly."""
    if duration <= 0:
        if relative:
            ctrl.move(x, y)
        else:
            ctrl.position = (x, y)
        return

    start_x, start_y = ctrl.position
    end_x = start_x + x if relative else x
    end_y = start_y + y if relative else y
    steps = max(1, int(duration / 0.01))
    step_sleep = duration / steps

    for i in range(1, steps + 1):
        t = i / steps
        eased = 1 - (1 - t) ** 3  # ease-out cubic
        ctrl.position = (
            int(start_x + (end_x - start_x) * eased),
            int(start_y + (end_y - start_y) * eased),
        )
        time.sleep(step_sleep)


@contextmanager
def _start_listener(listener: mouse.Listener | keyboard.Listener) -> Iterator[None]:
    """Start a pynput Listener non-blocking; ensure safe stop on exit."""
    listener.start()
    try:
        yield
    finally:
        if listener.running:
            listener.stop()
        listener.join(timeout=1.0)


# ============ 事件数据类 ============


@dataclass(slots=True)
class MouseEvent:
    """Mouse event record"""

    x: int
    y: int
    button: str | None = None
    pressed: bool = False
    scroll_dx: int = 0
    scroll_dy: int = 0
    event_type: Literal["move", "click", "scroll"] = "move"
    timestamp: float = field(default_factory=time.perf_counter)

    @override
    def __repr__(self) -> str:
        match self.event_type:
            case "move":
                return f"MouseEvent(move → ({self.x}, {self.y}))"
            case "click":
                action = "down" if self.pressed else "up"
                return f"MouseEvent({self.button} {action} @ ({self.x}, {self.y}))"
            case "scroll":
                return f"MouseEvent(scroll dx={self.scroll_dx} dy={self.scroll_dy} @ ({self.x}, {self.y}))"


@dataclass(slots=True)
class KeyboardEvent:
    """Keyboard event record"""

    key: str
    event_type: Literal["press", "release"]
    timestamp: float = field(default_factory=time.perf_counter)

    @override
    def __repr__(self) -> str:
        return f"KeyboardEvent({self.event_type} '{self.key}')"


# ============ Tkinter 对话框 ============


class _TkRoot:
    """Thread-safe tkinter root window (singleton)."""

    __slots__ = ("root", "_lock")
    _instance: _TkRoot | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.root: Any = None

    @classmethod
    def get(cls) -> _TkRoot:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def ensure(self) -> Any:
        with self._lock:
            import tkinter as tk

            if self.root is None:
                self.root = tk.Tk()
                self.root.withdraw()
            return self.root


def _center_window(dialog: Any) -> None:
    dialog.update_idletasks()
    w, h = dialog.winfo_width(), dialog.winfo_height()
    sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
    dialog.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")


def _run_tk_alert(title: str, text: str = "") -> None:
    try:
        import tkinter.messagebox as mb

        root = _TkRoot.get().ensure()
        root.attributes("-topmost", True)
        mb.showinfo(title, text, parent=root)
    except Exception:
        print(f"[alert] {title}: {text}")


def _run_tk_confirm(title: str, text: str, buttons: list[str]) -> str | None:
    try:
        import tkinter as tk

        root = _TkRoot.get().ensure()
        root.attributes("-topmost", True)
        result: list[str | None] = [None]

        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.transient(root)
        dialog.grab_set()

        tk.Label(
            dialog, text=text, wraplength=400, justify="left", padx=20, pady=15
        ).pack()
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=(0, 15), padx=20)
        for btn_text in buttons:
            tk.Button(
                btn_frame,
                text=btn_text,
                width=10,
                command=lambda v=btn_text: (result.__setitem__(0, v), dialog.destroy()),
            ).pack(side="left", padx=5)

        _center_window(dialog)
        dialog.focus_force()
        dialog.wait_window()
        return result[0]
    except Exception:
        print(f"[confirm] {title}: {text} → buttons={buttons}")
        return buttons[0]


def _run_tk_entry(
    title: str,
    text: str,
    default: str = "",
    *,
    password: bool = False,
    mask: str = "*",
) -> str | None:
    try:
        import tkinter as tk

        root = _TkRoot.get().ensure()
        root.attributes("-topmost", True)
        result: list[str | None] = [None]

        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.transient(root)
        dialog.grab_set()

        if text:
            tk.Label(
                dialog, text=text, wraplength=400, justify="left", padx=20, pady=15
            ).pack()

        entry = tk.Entry(dialog, width=40, show=mask if password else "")
        entry.insert(0, default)
        entry.select_range(0, "end")
        entry.pack(padx=20, pady=5, fill="x")

        def submit() -> None:
            val = entry.get()
            result[0] = val if val else None
            dialog.destroy()

        def cancel() -> None:
            result[0] = None
            dialog.destroy()

        entry.bind("<Return>", lambda _: submit())
        entry.bind("<Escape>", lambda _: cancel())
        entry.focus_set()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=(0, 15), padx=20)
        tk.Button(btn_frame, text="确定", width=10, command=submit).pack(
            side="left", padx=5
        )
        tk.Button(btn_frame, text="取消", width=10, command=cancel).pack(
            side="left", padx=5
        )

        _center_window(dialog)
        dialog.focus_force()
        dialog.wait_window()
        return result[0]
    except Exception:
        kind = "password" if password else "prompt"
        print(f"[{kind}] {title}: {text} (default={default!r})")
        return default or None


# ============ Auto ============


class Auto:
    """Top-level namespace for automation utilities."""

    class Mouse:
        """Mouse control and listening (pynput.mouse.Controller / Listener)."""

        @staticmethod
        def get_position() -> Point:
            pos = _mouse.position
            return int(pos[0]), int(pos[1])

        @staticmethod
        def move_to(x: int, y: int, duration: float = 0.0) -> None:
            """Move the mouse to absolute coordinates (uses ease-out if duration>0)."""
            _ease_move(_mouse, x, y, duration)
            _sleep()

        @staticmethod
        def move(dx: int, dy: int, duration: float = 0.0) -> None:
            """Move the mouse by a relative offset."""
            _ease_move(_mouse, dx, dy, duration, relative=True)
            _sleep()

        @staticmethod
        def click(
            x: int | None = None,
            y: int | None = None,
            button: str = "left",
            clicks: int = 1,
            interval: float = 0.0,
            duration: float = 0.0,
        ) -> None:
            """Click the mouse (uses pynput Controller.click)."""
            if x is not None and y is not None:
                Auto.Mouse.move_to(x, y, duration)
            btn = _resolve_button(button)
            n = max(1, clicks)
            if n == 1 or interval <= 0:
                _mouse.click(btn, n)
            else:
                for i in range(n):
                    _mouse.click(btn, 1)
                    if i < n - 1:
                        time.sleep(interval)
            _sleep()

        @staticmethod
        def double_click(
            x: int | None = None,
            y: int | None = None,
            button: str = "left",
            duration: float = 0.0,
        ) -> None:
            Auto.Mouse.click(
                x, y, button=button, clicks=2, interval=0.05, duration=duration
            )

        @staticmethod
        def mouse_down(button: str = "left") -> None:
            _mouse.press(_resolve_button(button))
            _sleep()

        @staticmethod
        def mouse_up(button: str = "left") -> None:
            _mouse.release(_resolve_button(button))
            _sleep()

        @staticmethod
        def scroll(dy: int, dx: int = 0) -> None:
            """Scroll the wheel (dy positive = up, dx positive = right)."""
            _mouse.scroll(dx, dy)
            _sleep()

        @staticmethod
        def drag(dx: int, dy: int, button: str = "left", duration: float = 0.5) -> None:
            """Drag by a relative offset."""
            btn = _resolve_button(button)
            _mouse.press(btn)
            try:
                _ease_move(_mouse, dx, dy, duration, relative=True)
            finally:
                _mouse.release(btn)
            _sleep()

        @staticmethod
        def drag_to(
            x: int, y: int, button: str = "left", duration: float = 0.5
        ) -> None:
            """Drag to an absolute position."""
            btn = _resolve_button(button)
            _mouse.press(btn)
            try:
                _ease_move(_mouse, x, y, duration)
            finally:
                _mouse.release(btn)
            _sleep()

        @staticmethod
        @contextmanager
        def listen(
            on_move: Callable[[MouseEvent], Any] | None = None,
            on_click: Callable[[MouseEvent], Any] | None = None,
            on_scroll: Callable[[MouseEvent], Any] | None = None,
            *,
            suppress: bool = False,
        ) -> Iterator[list[MouseEvent]]:
            """
            Mouse event listening context manager (non-blocking).

            Usage:
                with Auto.Mouse.listen(on_click=print) as events:
                    time.sleep(5)
            """
            events: list[MouseEvent] = []

            def _dispatch(
                evt: MouseEvent, cb: Callable[[MouseEvent], Any] | None
            ) -> None:
                events.append(evt)
                if cb:
                    cb(evt)

            def _on_move(x: int, y: int) -> None:
                _dispatch(MouseEvent(x=x, y=y, event_type="move"), on_move)

            def _on_click(x: int, y: int, button: MouseButton, pressed: bool) -> None:
                _dispatch(
                    MouseEvent(
                        x=x,
                        y=y,
                        button=_REVERSE_BUTTON_MAP.get(button, str(button)),
                        pressed=pressed,
                        event_type="click",
                    ),
                    on_click,
                )

            def _on_scroll(x: int, y: int, dx: int, dy: int) -> None:
                _dispatch(
                    MouseEvent(
                        x=x, y=y, scroll_dx=dx, scroll_dy=dy, event_type="scroll"
                    ),
                    on_scroll,
                )

            listener = mouse.Listener(
                on_move=_on_move,
                on_click=_on_click,
                on_scroll=_on_scroll,
                suppress=suppress,
            )
            with _start_listener(listener):
                yield events

    class KeyBoard:
        """Keyboard control and listening (pynput.keyboard.Controller / Listener)."""

        SPECIAL_KEYS = _SPECIAL_KEYS
        KEY_ALIASES = _KEY_ALIASES

        @staticmethod
        def write(text: str, interval: float = 0.0) -> None:
            """Simulate typing a string (pynput Controller.type)."""
            if interval <= 0:
                _kb.type(text)
            else:
                for i, ch in enumerate(text):
                    _kb.type(ch)
                    if i < len(text) - 1:
                        time.sleep(interval)
            _sleep()

        @staticmethod
        def press(key: KeyName, count: int = 1, interval: float = 0.05) -> None:
            """Press and release a key (pynput Controller.tap)."""
            resolved = _resolve_key(key)
            for i in range(max(1, count)):
                _kb.tap(resolved)
                if interval > 0 and i < count - 1:
                    time.sleep(interval)
            _sleep()

        @staticmethod
        def key_down(key: KeyName) -> None:
            _kb.press(_resolve_key(key))
            _sleep()

        @staticmethod
        def key_up(key: KeyName) -> None:
            _kb.release(_resolve_key(key))
            _sleep()

        @staticmethod
        def hotkey(*keys: KeyName, interval: float = 0.1) -> None:
            """Simulate hotkey combinations, e.g., KeyBoard.hotkey('ctrl', 'a')."""
            Auto.hotkey(*keys, interval=interval)

        @staticmethod
        @contextmanager
        def listen(
            on_press: Callable[[KeyboardEvent], Any] | None = None,
            on_release: Callable[[KeyboardEvent], Any] | None = None,
            *,
            suppress: bool = False,
        ) -> Iterator[list[KeyboardEvent]]:
            """
            Keyboard event listening context manager (non-blocking).

            Usage:
                with Auto.KeyBoard.listen(on_press=print) as events:
                    time.sleep(5)
            """
            events: list[KeyboardEvent] = []

            def _dispatch(
                key: Key | KeyCode | None, kind: Literal["press", "release"]
            ) -> None:
                evt = KeyboardEvent(key=_key_to_str(key), event_type=kind)
                events.append(evt)
                cb = on_press if kind == "press" else on_release
                if cb:
                    cb(evt)

            listener = keyboard.Listener(
                on_press=lambda k: _dispatch(k, "press"),
                on_release=lambda k: _dispatch(k, "release"),
                suppress=suppress,
            )
            with _start_listener(listener):
                yield events

        # Backwards compatibility
        _resolve_key = staticmethod(_resolve_key)
        _key_to_str = staticmethod(_key_to_str)

    class Msg:
        """Message dialogs (tkinter-based, pyautogui-like)."""

        @staticmethod
        def alert(title: str = "提示", text: str = "", button: str = "ok") -> str:
            _run_tk_alert(title, text)
            return button

        @staticmethod
        def confirm(
            title: str = "确认",
            text: str = "",
            buttons: list[str] | None = None,
        ) -> str | None:
            return _run_tk_confirm(title, text, buttons or ["ok", "cancel"])

        @staticmethod
        def prompt(
            title: str = "输入", text: str = "", default: str = ""
        ) -> str | None:
            return _run_tk_entry(title, text, default)

        @staticmethod
        def password(
            title: str = "密码",
            text: str = "",
            default: str = "",
            mask: str = "*",
        ) -> str | None:
            return _run_tk_entry(title, text, default, password=True, mask=mask)

    @staticmethod
    def hotkey(*keys: KeyName, interval: float = 0.1) -> None:
        """
        Simulate hotkeys, e.g., Auto.hotkey('win', 'd') to show the desktop.
        Implemented using pynput Controller.pressed + tap.
        """
        resolved = [_resolve_key(k) for k in keys]
        if not resolved:
            return

        ctx = _kb.pressed(*resolved[:-1]) if len(resolved) > 1 else nullcontext()
        with ctx:
            _kb.tap(resolved[-1])
        _sleep(interval)

    @staticmethod
    def show_msg(msg: str, title: str = "提示") -> None:
        _run_tk_alert(title, msg)


__all__ = [
    "Auto",
    "KeyboardEvent",
    "MouseEvent",
    "get_pause",
    "set_pause",
]
