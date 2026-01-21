# output_manager.py

import hashlib
import os
import re

from numclass.fmt import strip_ansi
from numclass.utility import get_terminal_width
from numclass.workspace import workspace_dir

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._=-]+")


def resolve_output_path(path: str, workspace_root: str) -> str:
    """
    Resolve user-provided output path.

    Rules:
    - '~' expanded to user home
    - Absolute paths unchanged
    - Relative paths are relative to workspace_root
    """
    if not path:
        raise ValueError("Output path is empty")

    # Expand ~ (Linux/macOS + works on Windows too)
    path = os.path.expanduser(path)

    # Absolute path → keep
    if os.path.isabs(path):
        return os.path.normpath(path)

    # Relative path → workspace-relative
    return os.path.normpath(os.path.join(workspace_root, path))


def _next_available_path(path: str) -> str:
    """
    If `path` does not exist, return it.
    Otherwise return path with _2, _3, ... inserted before the extension.
    """
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    i = 2
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _fallback_filename_for_number(n: int, ext: str = ".txt", head: int = 12, tail: int = 12) -> str:
    """
    Filesystem-safe shortened filename that still lets you recognize the number.

    Format:
      digits=<ndigits>_<sign><head>...<tail>_sha<sha12>.txt

    - ndigits counts decimal digits excluding '-'
    - sign is '-' if n is negative, otherwise empty
    - head/tail are 12 digits each (configurable)
    - sha12 = first 12 hex chars of sha256(str(n))
    """
    s = str(n)  # keep '-' if present
    sign = "-" if s.startswith("-") else ""
    s_abs = s[1:] if sign else s
    nd = len(s_abs)

    sha12 = hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

    # Always keep exactly 12 before and after "..." (as requested).
    # If the number is shorter than 12 digits, just use what we have (still safe).
    head_part = s_abs[:head]
    tail_part = s_abs[-tail:] if len(s_abs) > tail else s_abs

    stem = f"digits={nd}_{sign}{head_part}...{tail_part}_sha={sha12}"
    stem = _SAFE_CHARS_RE.sub("_", stem).strip("._-=")
    return stem + ext


def _choose_split_output_path(directory: str, number: int, ext: str = ".txt", max_full_path_len: int = 250) -> str:
    """
    Use '<number>.txt' when the FULL path length is < max_full_path_len.
    Otherwise use a shortened, recognizable name.
    """
    # Candidate 1: original behavior (keep '-' sign)
    original = os.path.join(directory, f"{number}{ext}")
    if len(original) < max_full_path_len:
        return original

    # Candidate 2: safe fallback
    return os.path.join(directory, _fallback_filename_for_number(number, ext=ext))


class OutputManager:
    """
    Handles all printing/output, including to screen and/or file.

    Usage:
        # Split mode (per-number file):
        om = OutputManager(output_file="results/", number=42)
        om.write("Hello")   # prints and buffers; file written on close()
        om.close()

        # Single file (append all runs to one file):
        om = OutputManager(output_file="results/all.txt")
        om.write("Hello")   # prints and appends safely via a context manager
        om.close()
    """

    def __init__(self, output_file: str | None = None, quiet: bool = False, number: int | None = None):
        """
        Parameters:
            output_file:
                None or ""       => screen only
                "." or "./"      => per-number files in current dir
                endswith "/"     => per-number files in specified dir
                path/to/file.txt => append all runs to this file
            quiet: if True, no output to screen (only to file)
            number: integer, used for filename in per-number mode
        """
        self.quiet = quiet
        self.output_file = output_file or ""
        self.number = number
        self._buffer: list[str] = []

        # Resolve mode & paths; no files are opened here (SIM115-friendly).
        self._mode: str = "none"     # "none" | "split" | "single"
        self._split_path: str | None = None
        self._single_path: str | None = None

        if self.output_file in (".", "./") or self.output_file.endswith("/"):
            if number is None:
                raise ValueError("A number must be provided when outputting to a directory.")
            # Use workspace folder as default (relative) output folder
            workspace = workspace_dir()
            directory = resolve_output_path(self.output_file, workspace)
            if not directory.endswith("/"):
                directory += "/"
            if directory not in (".", "./"):
                os.makedirs(directory, exist_ok=True)

                self._mode = "split"
                self._split_path = _choose_split_output_path(directory, number, ext=".txt", max_full_path_len=250)

        elif self.output_file:
            # Single file append mode
            workspace = workspace_dir()
            path = resolve_output_path(self.output_file, workspace)

            # Ensure parent folder exists (important for workspace-relative paths like "logs/out.txt")
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            self._mode = "single"
            self._single_path = path

    def clear_line(self) -> None:
        """Erase the current terminal line for in-place progress updates."""
        if self.quiet:
            return
        try:
            # Fast ANSI path (works on modern Windows if Colorama/VT enabled)
            self.write_screen("\r\x1b[2K", end="", flush=True)
        except Exception:
            # Fallback: overwrite the line with spaces
            width = get_terminal_width()
            self.write_screen("\r" + " " * width + "\r", end="", flush=True)

    def write(self, *args, sep: str = " ", end: str = "\n") -> None:
        """Write to screen and file (if configured)."""
        text = sep.join(str(a) for a in args) + end
        self._buffer.append(text)

        # Screen
        if not self.quiet:
            print(text, end="")

        # File handling (SIM115-safe: always use a context manager)
        if self._mode == "single" and self._single_path:
            # Append per call
            try:
                with open(self._single_path, "a", encoding="utf-8") as fh:
                    fh.write(strip_ansi(text))
            except Exception:
                # Swallow file errors to preserve CLI UX; caller can inspect stdout buffer via getvalue()
                pass
        # For "split" mode we defer writing until close(), so we don't truncate on each call.

    def write_screen(self, *args, sep: str = " ", end: str = "\n", flush: bool = True) -> None:
        """Write only to the screen, never to the file."""
        if self.quiet:
            return
        print(*args, sep=sep, end=end, flush=flush)

    def getvalue(self) -> str:
        """Returns everything printed (with color codes)."""
        return "".join(self._buffer)

    def close(self) -> None:
        """Flush buffered output to per-number file (split mode) and add separator in single-file mode."""
        # Split mode: write once
        if self._mode == "split" and self._split_path and self._buffer:
            try:
                content = strip_ansi("".join(self._buffer))
                with open(self._split_path, "w", encoding="utf-8") as fh:
                    fh.write(content)
            except Exception as e:
                if not self.quiet:
                    self.write_screen(f"[WARNING] Could not write output file: {self._split_path} ({type(e).__name__}: {e})")
            return

        # Single mode: add separator between runs
        if self._mode == "single" and self._single_path and self._buffer:
            try:
                with open(self._single_path, "a", encoding="utf-8") as fh:
                    fh.write("\n")  # one empty line between runs
            except Exception:
                pass

    def __del__(self):
        # Best-effort flush for split mode if the user forgot to call close()
        try:
            self.close()
        except Exception:
            pass
