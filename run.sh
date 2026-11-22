#!/usr/bin/env bash
# One-step setup and start of the iiwa app. The first run creates the Python
# environment and downloads the robot model; later runs start right away.
#
#   ./run.sh                 start the app and open the browser
#   ./run.sh --local         serve on localhost only (no Tailscale)
#   ./run.sh --port 8081     other arguments go to iiwa_viser.py
#   NO_BROWSER=1 ./run.sh    don't open the browser (e.g. over SSH)
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# Keep the venv isolated: a sourced ROS (or other) setup exports PYTHONPATH,
# which would leak system packages into it.
unset PYTHONPATH

VENV=.venv
PY="$VENV/bin/python"
MODEL_DIR=third_party/mujoco_menagerie
MODEL="$MODEL_DIR/kuka_iiwa_14/scene.xml"
# Fixed Menagerie commit, so every install gets the same robot model.
MENAGERIE_COMMIT=370c43965323c0944b433b202aac5058b5f907a9

die() { echo "error: $*" >&2; exit 1; }

# 0. Prerequisites.
command -v python3 >/dev/null || die "python3 not found. Install Python 3.10 or newer."
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
    die "Python 3.10 or newer needed, found $(python3 --version 2>&1)."
command -v git >/dev/null || die "git not found (needed once, to download the robot model)."

# 1. Virtual environment. Ubuntu often lacks python3-venv: fall back to virtualenv.
if [[ ! -x "$PY" ]]; then
    echo "==> Creating the Python environment in $VENV"
    rm -rf "$VENV"
    if ! python3 -m venv "$VENV" >/dev/null 2>&1; then
        rm -rf "$VENV"
        python3 -m virtualenv --version >/dev/null 2>&1 || python3 -m pip install --user -q virtualenv ||
            die "could not create a virtual environment. On Ubuntu: sudo apt install python3-venv"
        python3 -m virtualenv -q "$VENV"
    fi
fi

# 2. Packages: (re)installed only when requirements.txt changed.
hash_file() { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; }
REQ_HASH=$(hash_file requirements.txt | cut -d' ' -f1)
if [[ "$(cat "$VENV/.requirements.sha256" 2>/dev/null)" != "$REQ_HASH" ]]; then
    echo "==> Installing Python packages (first run: a minute or two)"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q -r requirements.txt
    echo "$REQ_HASH" > "$VENV/.requirements.sha256"
fi

# 3. Robot model: only the iiwa14 folder of MuJoCo Menagerie (~11 MB).
if [[ ! -f "$MODEL" ]]; then
    echo "==> Downloading the iiwa14 robot model"
    rm -rf "$MODEL_DIR"
    git init -q "$MODEL_DIR"
    git -C "$MODEL_DIR" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
    git -C "$MODEL_DIR" sparse-checkout set kuka_iiwa_14
    git -C "$MODEL_DIR" fetch -q --depth 1 --filter=blob:none origin "$MENAGERIE_COMMIT" ||
        die "model download failed (no internet?). Run ./run.sh again to retry."
    git -C "$MODEL_DIR" checkout -q FETCH_HEAD
fi

# 4. Run (exec: Ctrl+C goes straight to the app, which exits cleanly). The app
# opens the browser itself, on the URL it actually serves (Tailscale or localhost).
OPEN=(--open)
[[ -n "${NO_BROWSER:-}" ]] && OPEN=()
exec "$PY" iiwa_viser.py "${OPEN[@]}" "$@"
