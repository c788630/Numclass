# src/numclass/cli.py

"""
Number Classifier - Mathematical Classifications and Curiosities

Author:      Marcel M W van Dinteren <m.vandinteren1@chello.nl>
Date:        2025-12-25
Version:     2.0.0

Description:
    Classifies an integer n according to various mathematical properties
    and curiosities. Outputs classification, statistics, and special sequences.

usage: see numclass.py -h

License:
    Code: CC BY-NC 4.0 - © 2024 Marcel M W van Dinteren
          https://creativecommons.org/licenses/by-nc/4.0/
    Data: CC BY-NC 3.0 - From OEIS Foundation Inc.
          https://oeis.org/copyright.html
          - Data from OEIS A004394 (https://oeis.org/A004394/b004394.txt)
          - Data from OEIS A104272 (https://oeis.org/A104272/b104272.txt)
          - Modified: shortened and/or with comments.

"""

from __future__ import annotations

import argparse
import faulthandler
import os
import platform
import sys
import textwrap
import threading
import time
import traceback
from importlib.resources import files as pkg_files
from typing import NamedTuple

from colorama import Fore, Style
from colorama import init as colorama_init

from numclass.__init__ import __version__ as _ver
from numclass.classify import classify
from numclass.dataio import data_path
from numclass.display import (
    print_classifications,
    print_profiles_with_descriptions,
    print_statistics,
    show_classifier_list,
    show_intro_help,
)
from numclass.expreval import SpecialInputHandled, _parse_int_or_expr
from numclass.output_manager import OutputManager
from numclass.registry import (  # may raise if not present in older versions
    discover,
    discover_with_report,
)
from numclass.runtime import APPLY, CFG, ensure_runtime_deps
from numclass.runtime import current as _rt_current
from numclass.utility import (
    UserInputError,
    build_ctx,
    clear_screen,
    flatten_dotted,
    get_terminal_height,
    get_terminal_width,
    typename,
    validate_output_setting,
)
from numclass.workspace import ensure_workspace_seeded, seed_workspace, workspace_dir


# In memory session history
class HistoryItem(NamedTuple):
    n: int
    profile: str | None
    timestamp: float


_HISTORY: list[HistoryItem] = []


def add_to_history(n: int, profile: str | None = None) -> None:
    _HISTORY.append(HistoryItem(n=n, profile=profile, timestamp=time.time()))


def get_history() -> list[HistoryItem]:
    return list(_HISTORY)


def clear_history() -> None:  # not yet used, for future expansion
    _HISTORY.clear()


def _install_loud_error_handlers(debug: bool) -> None:
    if not debug:
        return
    # Always show full Python tracebacks
    faulthandler.enable()

    def _excepthook(exc_type, exc, tb):
        sys.stderr.write("\n[UNCAUGHT EXCEPTION]\n")
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
        sys.stderr.flush()
    sys.excepthook = _excepthook

    # Also catch exceptions in threads (Py3.8+)
    def _thread_excepthook(args):
        sys.stderr.write("\n[UNCAUGHT THREAD EXCEPTION]\n")
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)
        sys.stderr.flush()
    try:
        threading.excepthook = _thread_excepthook  # may not exist on very old Python
    except Exception:
        pass


def _print_user_error(msg: str) -> None:
    """Uniform, one-line friendly error."""
    # If the messages already start with "Invalid input:", keep that; otherwise prefix:
    prefix = f"{Fore.RED}Error:{Style.RESET_ALL}"
    if not (msg.startswith("Invalid input:") or msg.startswith("Error:")):
        msg = f"{prefix} {msg}"
    print(msg, file=sys.stderr)


def _resolve_inputs(items: list[str]) -> tuple[str | None, int | None]:
    """Return (profile, number) based on the first two positionals.

    Rules:
      - If one item parses as int/expr -> number; else -> profile
      - If two items:
          * first numeric, second not -> (None, number)
          * first not, second numeric -> (profile, number)
          * both numeric -> take the first as number (old behavior)
          * neither numeric -> (profile, None)
    """
    if not items:
        return None, None

    if len(items) == 1:
        try:
            n = _parse_int_or_expr(items[0])
        except SpecialInputHandled as e:
            print(e.output)
            return None, None
        return (None, n) if n is not None else (items[0], None)

    a, b = items[0], items[1]
    na, nb = _parse_int_or_expr(a), _parse_int_or_expr(b)

    if na is not None and nb is None:
        return None, na
    if na is None and nb is not None:
        return a, nb
    if na is not None and nb is not None:
        return None, na
    return a, None


def _select_profile_name(args, CONFIG):
    """
    Precedence:
      1) explicit --profile
      2) last used (from workspace)
      3) 'default'
    """
    name = getattr(args, "profile", None)
    if name:
        return name
    try:
        last = CONFIG.read_current_profile() if CONFIG else None
    except Exception:
        last = None
    return last or "default"


# ---- argparse ----
def _build_parser() -> argparse.ArgumentParser:

    epilog = textwrap.dedent("""\
    commands:
      init
          Create workspace folders and copy packaged sample files if missing.

      init overwrite
          Meant for developers. Requires environment variable NUMCLASS_DEV=1.
          Copies all initial classifiers, datafiles and profiles.
          Your personal edits in the Windows Documents folders will be lost!

      list
          List all available classifiers.

      where
          Show the workspace and package paths.
    """)

    p = argparse.ArgumentParser(
        description="Number Classifier — mathematical classifications & curiosities",
        usage=(
            "numclass [[profile] [integer]] [--output OUTPUT] [--quiet] [--no-details] [--debug]\n"
            "       numclass -h | --help\n"
            "       numclass --init [--overwrite]\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("items", nargs="*", metavar="[[profile] integer]]",
                   help="optional profile name followed by an integer to classify")
    p.add_argument("--output", default=None, help="Write results to a file (also prints unless --quiet)")
    p.add_argument("--quiet", action="store_true", help="Suppress live progress and summary output")
    p.add_argument("--no-details", action="store_true", help="Omit explanations and classifier details")
    p.add_argument("--debug", action="store_true", help="Show per-classifier timings and internal trace info")

    return p


def main(argv=None) -> int:
    """Thin wrapper: catch friendly errors, hide tracebacks unless debug."""
    try:
        return _main_impl(argv)
    except UserInputError as e:
        _print_user_error(str(e))
        return 2
    except KeyboardInterrupt:
        print("Aborted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        # Only show traceback in debug mode or if flag enabled
        debug = ("--debug" in (argv or sys.argv))
        if debug:
            raise
        print(f"Unexpected error: {e.__class__.__name__}: {e}", file=sys.stderr)
        print("Run with --debug for a full traceback.", file=sys.stderr)
        return 1


# ---- main ----
def _main_impl(argv=None) -> int:

    colorama_init(autoreset=True)

    def configure_text_streams():
        # 1) Respect explicit user choice
        if os.environ.get("PYTHONIOENCODING"):
            return  # Python is already using this

        try:
            # Only touch redirected output (pipes/files), leave TTY as-is
            if not sys.stdout.isatty():
                enc = (sys.stdout.encoding or "").lower()

                if platform.system() == "Windows":
                    # 2) Windows: force UTF-8 for redirected output
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
                # 3) POSIX: fix only if clearly unsafe (ASCII)
                elif enc in ("", "ascii", "us-ascii"):
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Be fail-safe: never crash because of reconfigure
            pass

    configure_text_streams()

    # config (profile I/O)
    try:
        import numclass.config as CONFIG
    except Exception:
        CONFIG = None

    parser = _build_parser()
    args = parser.parse_args(argv)
    rt = _rt_current()
    rt.debug = bool(args.debug)

    _install_loud_error_handlers(args.debug)

    if getattr(args, "overwrite", False) and not getattr(args, "init", False):
        parser.error("--overwrite can only be used together with --init")

    if not ensure_runtime_deps(strict=True):
        return 1

    # Ensure a first-run workspace seed silently
    ws, seeded, copied = ensure_workspace_seeded()

    # --- discovery (atomic + intersections on Index) ---

    if args.debug:
        print()
        print(f"Terminal {get_terminal_width()}x{get_terminal_height()}")
        index, rep = discover_with_report(workspace_dir())
        # atomic summary
        print(f"[debug] discovered atomic: {len(index.funcs)}", file=sys.stderr)
        # workspace modules
        for name, cnt in rep.ws_loaded:
            print(f"[discovery] {Fore.GREEN}ws OK{Style.RESET_ALL} {name}: {cnt} label(s)", file=sys.stderr)
        for name, err in rep.ws_failed:
            print(f"[discovery] {Fore.RED}ws FAIL{Style.RESET_ALL} {name}: {err}", file=sys.stderr)
        count = len(rep.skipped_duplicates)  # if it's a list/tuple
        if count:
            print(
                f"[discovery] {Fore.YELLOW}SKIP{Style.RESET_ALL} {count} duplicate label(s) skipped.",
                file=sys.stderr
            )

        # intersections count
        print(f"[discovery] intersections OK: {len(index.intersections)} from {data_path('intersections.toml')}", file=sys.stderr)
    else:
        index = discover(workspace_dir())

    # --- parse inputs: command or profile + number ---
    profile, smart_n = _resolve_inputs(args.items)

    if profile == "active":
        print(f"Active profile: {CONFIG.read_current_profile()}")
        return 0
    _TWO_ARGS = 2
    if profile == "init":
        if len(args.items) == _TWO_ARGS and args.items[1] == "overwrite":
            if os.environ.get("NUMCLASS_DEV") != "1":
                print("Refusing to overwrite: set NUMCLASS_DEV=1 to enable developer overwrite.")
                return 2
            ws, copied = seed_workspace(overwrite=True)  # overwrite all three: profiles, data, classifiers
            print(f"Workspace ready at: {ws} (overwrote existing files)")
            print(f"Copied -> profiles: {copied.get('profiles', 0)}, "
                  f"data: {copied.get('data', 0)}, classifiers: {copied.get('classifiers', 0)}")
            return 0

        ws, seeded, copied = ensure_workspace_seeded()  # copy-if-missing
        print(f"Workspace ready at: {ws}")
        print(f"Copied -> profiles: {copied.get('profiles', 0)}, data: {copied.get('data', 0)}, "
              f"classifiers: {copied.get('classifiers', 0)}")
        return 0

    if profile == "list":
        show_classifier_list(index, paged=False)
        return 0
    if profile == "where":
        print(f"Workspace: {workspace_dir()}")
        print(f"Package:   {pkg_files('numclass').joinpath('..').resolve()}")
        return 0

    # Choose profile: explicit → last-used → default
    def _select_profile_name(explicit: str | None) -> str:
        if explicit:
            return explicit
        if CONFIG:
            try:
                last = CONFIG.read_current_profile()
            except Exception:
                last = None
            if last and CONFIG.has_profile(last):
                return last
        return "default"

    profile_name = _select_profile_name(profile)

    # Validate explicit profile (if given)
    if CONFIG and profile and not CONFIG.has_profile(profile):
        print(f"Unknown profile: '{profile}'")
        try:
            print("Available profiles:", ", ".join(CONFIG.list_all_profiles()))
        except Exception:
            pass
        return 2

    # Load & apply profile (with fallback if missing)
    if CONFIG:
        try:
            if not CONFIG.has_profile(profile_name):
                profile_name = "default"
            selected = CONFIG.load_settings(profile_name)
            APPLY(selected)  # install into runtime

            # Effective limit from profile (fallback = 100_000)
            limit = int(CFG("BEHAVIOUR.MAX_DIGITS", 100_000))
            if not os.environ.get("PYTHONINTMAXSTRDIGITS"):
                try:
                    sys.set_int_max_str_digits(limit)
                except Exception:
                    pass

            if args.debug:
                # Header
                print(f"[debug] active profile: {profile_name}", file=sys.stderr)
                src_path = getattr(selected, "_source", None)
                if src_path:
                    print(f"[debug] profile file: {src_path}", file=sys.stderr)

                # Load raw TOML so we can list the keys as written in the profile
                raw: dict = {}
                _toml = None
                try:
                    import tomllib as _toml  # py311+
                except Exception:
                    try:
                        import tomli as _toml  # fallback
                    except Exception:
                        _toml = None
                if src_path and _toml:
                    try:
                        with open(src_path, "rb") as f:
                            raw = _toml.load(f)
                    except Exception as e:
                        print(f"[debug] (warn) failed reading TOML: {e}", file=sys.stderr)

                # Print settings list (raw keys → runtime value + type)
                if raw:
                    print("[debug] profile keys (raw → runtime value/type):", file=sys.stderr)
                    flat = flatten_dotted(raw)
                    for k in sorted(flat.keys(), key=str.lower):
                        runtime_val = CFG(k, None)
                        print(f"        {k:.<50} {runtime_val!r} ({typename(runtime_val)})", file=sys.stderr)
                else:
                    # Fallback: dump everything currently applied (flattened)

                    flat_rt = flatten_dotted(_rt_current().settings)
                    print("[debug] runtime settings (flattened):", file=sys.stderr)
                    for k in sorted(flat_rt.keys(), key=str.lower):
                        v = flat_rt[k]
                        print(f"  - {k}: {v!r} ({typename(v)})", file=sys.stderr)

                # spacer
                print(file=sys.stderr)
        except UserInputError as e:
            _print_user_error(str(e))   # clean, no traceback
            return 2
        except Exception as e:
            print(f"Failed to load profile: {e}\n", file=sys.stderr)
            # continue with defaults

    # --- output routing (CLI --output overrides profile OUTPUT_FILE) ---
    try:
        _CLI_OUTPUT_TARGET = validate_output_setting(args.output)  # None => use profile OUTPUT_FILE
    except ValueError as e:
        print(f"Fatal error in --output: {e}", file=sys.stderr)
        return 1

    def make_output_manager(n=None) -> OutputManager:
        # read OUTPUT_FILE from runtime each time so profile switches take effect
        target = _CLI_OUTPUT_TARGET if _CLI_OUTPUT_TARGET is not None else CFG("OUTPUT.OUTPUT_FILE", None)
        return OutputManager(output_file=target, quiet=args.quiet, number=n)

    # --- one-shot number path ---
    if smart_n is not None:
        n = smart_n
        om = make_output_manager(n)
        try:

            ctx_for_n = build_ctx(abs(n))

            # 1) number & divisor stats
            print_statistics(n, args.items[0], show_details=not args.no_details, om=om)

            # 2) run classifiers (atomic + intersections)
            results = classify(n, index=index, progress=True, ctx=ctx_for_n)

            # 3) render results
            print_classifications(n, results, show_details=not args.no_details, om=om, index=index)
        finally:
            om.close()
        return 0

    # --- REPL ---
    if not _rt_current().debug:
        clear_screen()
    print(f"{Fore.YELLOW}{Style.BRIGHT}Number Classifier v{_ver} — Mathematical Classifications & Curiosities{Style.RESET_ALL}")

    current_profile = profile_name
    while True:
        try:
            prompt = f"\nProfile: {current_profile} — Enter an integer, command or profile (h=Help, q=Quit): "
            user_input = input(prompt).strip()

            low = user_input.lower()
            if low in {"", "q", "quit"}:
                break

            if low in {"h", "help"}:
                om_help = OutputManager(output_file=None, quiet=False, number="help")
                try:
                    show_intro_help(index, om=om_help)
                finally:
                    om_help.close()
                continue

            if low in {"p", "list profiles"}:
                print_profiles_with_descriptions()
                continue

            if low in {"hist", "history"}:
                hist = get_history()
                if not hist:
                    print("History is empty.")
                    continue
                else:
                    for item in hist:
                        ts = time.strftime("%H:%M:%S", time.localtime(item.timestamp))
                        print(f"{ts}  n={item.n:<15}  profile={item.profile or '-'}")
                    continue

            if low.startswith("fast"):
                parts = low.split()
                rt = _rt_current()
                if len(parts) == 1 or parts[1] == "status":
                    state = "ON" if rt.fast_mode else "OFF"
                    print(f"Fast mode is currently {state}.")
                elif parts[1] == "on":
                    rt.fast_mode = True
                    print("Fast mode ENABLED: slow/expensive operations may be skipped.")
                elif parts[1] == "off":
                    rt.fast_mode = False
                    print("Fast mode DISABLED: slow operations allowed up to per-classifier limits.")
                else:
                    print("Usage: FAST [on|off|status]")
                continue

            if low.startswith("debug"):
                parts = low.split()
                rt = _rt_current()
                if len(parts) == 1 or parts[1] in {"status"}:
                    state = "ON" if rt.debug else "OFF"
                    print(f"Debug is currently {state}.")
                elif parts[1] == "on":
                    rt.debug = True
                    print("Debug mode enabled for this session.")
                elif parts[1] == "off":
                    rt.debug = False
                    print("Debug mode disabled for this session.")
                else:
                    print("Usage: DEBUG [on|off|status]")
                continue

            # number?
            try:
                n = _parse_int_or_expr(user_input)

            except SpecialInputHandled as e:
                # Special handler output (Mersenne / easter eggs)
                print(e.output)
                continue

            except UserInputError as e:
                # Nicely formatted input error in red, no traceback
                msg = str(e)
                prefix = f"{Fore.RED}Invalid input:{Style.RESET_ALL}"
                msg = msg.replace("Invalid input:", prefix, 1) if msg.startswith("Invalid input:") else f"{prefix} {msg}"
                print(msg, file=sys.stderr)
                continue

            if n is not None:
                om = None
                try:
                    om = make_output_manager(n)
                    ctx_for_n = print_statistics(n, user_input, show_details=not args.no_details, om=om)
                    results = classify(n, index=index, progress=True, ctx=ctx_for_n)
                    print_classifications(n, results, show_details=not args.no_details, om=om, index=index)
                    add_to_history(n, current_profile)

                except UserInputError as e:
                    # Keep this in case classifiers or other code raise UserInputError
                    msg = str(e)
                    prefix = f"{Fore.RED}Invalid input:{Style.RESET_ALL}"
                    msg = msg.replace("Invalid input:", prefix, 1) if msg.startswith("Invalid input:") else f"{prefix} {msg}"
                    print(msg, file=sys.stderr)

                finally:
                    if om is not None:
                        om.close()
                continue

            # treat as profile switch
            if CONFIG and CONFIG.has_profile(user_input):
                try:
                    selected = CONFIG.load_settings(user_input)
                    APPLY(selected)  # apply

                    try:
                        CONFIG.write_current_profile(user_input)  # remember
                    except Exception:
                        pass
                    current_profile = user_input
                    print(f"Applied profile: {current_profile}")
                except Exception as e:
                    print(f"{Fore.RED}Failed to load profile {Style.RESET_ALL}'{user_input}': {e}", file=sys.stderr)
                continue

            print(f"{Fore.RED}Invalid input: {Style.RESET_ALL}'{user_input}'. Type H for help.")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        except Exception as e:
            if _rt_current().debug:
                traceback.print_exc()
            else:
                _print_user_error(f"{e.__class__.__name__}: {e}")
            continue


if __name__ == "__main__":
    raise SystemExit(main())
