# output_manager.py

import os
from utility import strip_ansi


class OutputManager:
    """
    Handles all printing/output, including to screen and/or file.

    Usage:
        om = OutputManager(output_file="results/", number=42)
        om.write("Hello")  # prints and writes to file (results/42.txt)
    """
    def __init__(self, output_file=None, quiet=False, number=None):
        """
        Parameters:
            output_file: None or "" => screen only
                         "." or "./" => per-number files in current dir
                         endswith "/" => per-number files in specified dir
                         path/to/file.txt => append all runs to this file
            quiet: if True, no output to screen (only to file)
            number: integer, used for filename in per-number mode
        """
        self.quiet = quiet
        self.output_file = output_file
        self.number = number
        self._buffer = []

        self._file_handle = None  # Always defined

        # Decide file writing mode
        if output_file and output_file not in ("", None):
            if output_file in (".", "./") or output_file.endswith("/"):
                # Per-number file in current or specified dir
                if number is None:
                    raise ValueError("A number must be provided when "
                                     "outputting to a directory.")
                directory = output_file
                if not directory.endswith("/"):
                    directory += "/"
                if directory not in (".", "./"):
                    os.makedirs(directory, exist_ok=True)
                filename = f"{directory}{number}.txt"
                self._file_handle = open(filename, "w", encoding="utf-8")
            else:
                # Single (append) file
                dirpart = os.path.dirname(output_file)
                if dirpart and not os.path.exists(dirpart):
                    os.makedirs(dirpart, exist_ok=True)
                self._file_handle = open(output_file, "a", encoding="utf-8")

    def write(self, *args, sep=" ", end="\n"):
        """Write to screen and file."""
        text = sep.join(str(a) for a in args) + end
        self._buffer.append(text)
        if not self.quiet:
            print(text, end="")  # Don't double end!
        if self._file_handle:
            self._file_handle.write(strip_ansi(text))
            self._file_handle.flush()

    def write_screen(self, *args, sep=" ", end="\n", flush=True):
        """
        Write only to the screen, never to the file.
        """
        print(*args, sep=sep, end=end)

    def getvalue(self):
        """Returns everything printed (with color codes)."""
        return "".join(self._buffer)

    def close(self):
        """Closes the output file if open."""
        if hasattr(self, '_file_handle') and self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        self.close()
