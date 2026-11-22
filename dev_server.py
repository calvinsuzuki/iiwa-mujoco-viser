"""Development runner: starts iiwa_viser.py and restarts it whenever a source file changes.

Run:  .venv/bin/python dev_server.py [any iiwa_viser.py args, e.g. --port 8080]
Watches *.py in this folder and formulas/*.svg. Open browser tabs reconnect on their own.
"""

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PATTERNS = ("*.py", "formulas/*.svg")


def snapshot():
    files = (p for pat in PATTERNS for p in HERE.glob(pat))
    return {p: p.stat().st_mtime for p in files if p.name != "dev_server.py"}


def start():
    print("[dev] starting iiwa_viser.py", flush=True)
    return subprocess.Popen([sys.executable, "-u", str(HERE / "iiwa_viser.py"), *sys.argv[1:]], cwd=HERE)


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    seen, proc = snapshot(), start()
    try:
        while True:
            time.sleep(0.5)
            now = snapshot()
            if now != seen:
                changed = sorted(p.name for p in now.keys() | seen.keys() if now.get(p) != seen.get(p))
                time.sleep(0.3)                    # let the editor finish writing
                seen = snapshot()
                print(f"[dev] changed: {', '.join(changed)} -> restarting", flush=True)
                stop(proc)
                proc = start()
            elif proc.poll() is not None:          # crashed: wait for the next edit
                print(f"[dev] exited with code {proc.returncode}; waiting for a change", flush=True)
                while snapshot() == seen:
                    time.sleep(0.5)
                seen = snapshot()
                proc = start()
    except KeyboardInterrupt:
        pass
    finally:
        stop(proc)


if __name__ == "__main__":
    main()
