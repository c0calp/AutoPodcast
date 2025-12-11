# autopodcast/audio_preprocessing.py

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .models import AudioFileInfo
from shutil import which

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# We hardcode the absolute paths to your ffmpeg and ffprobe executables.
# If you ever move them, update these two paths.
# You can still override them with environment variables FFMPEG_BIN/FFPROBE_BIN.
# ---------------------------------------------------------------------------

DEFAULT_FFMPEG_PATH = which("ffmpeg")
DEFAULT_FFPROBE_PATH = which("ffprobe")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", DEFAULT_FFMPEG_PATH)
FFPROBE_BIN = os.getenv("FFPROBE_BIN", DEFAULT_FFPROBE_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_executable(name_or_path: str) -> str:
    """
    Resolve either:
    - an absolute/relative path to an .exe, or
    - a bare command name like 'ffmpeg' using PATH.

    Raises FileNotFoundError with a detailed message if not found.
    """
    p = Path(name_or_path)

    # Case 1: it's an actual existing file (absolute or relative path)
    if p.is_file():
        return str(p)

    # Case 2: use PATH (ffmpeg/ffprobe installed globally)
    found = shutil.which(name_or_path)
    if found:
        return found

    # Nothing worked → very clear error
    raise FileNotFoundError(
        f"Could not find executable '{name_or_path}'.\n"
        f"Tried as a direct path: {p}\n"
        f"And searched in PATH: {os.environ.get('PATH', '')}\n"
        f"Make sure ffmpeg/ffprobe are installed and visible to this Python process.\n"
        f"You can also set FFMPEG_BIN / FFPROBE_BIN environment variables or\n"
        f"update DEFAULT_FFMPEG_PATH / DEFAULT_FFPROBE_PATH in audio_preprocessing.py."
    )


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    """
    Run a subprocess command and raise if it fails.
    We also resolve the first element (the executable) so we avoid WinError 2.
    """
    if not cmd:
        raise ValueError("Empty command list passed to _run_cmd.")

    # Resolve the executable (cmd[0]) to a real path or PATH entry
    exe = _resolve_executable(cmd[0])
    full_cmd = [exe, *cmd[1:]]

    # Helpful debug print so you can see exactly what is being run
    print("Running external command:", full_cmd)

    try:
        proc = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return proc
    except FileNotFoundError as e:
        # This is the classic WinError 2 case
        raise FileNotFoundError(
            f"Failed to run external command.\n"
            f"Command: {full_cmd}\n"
            f"Original error: {e}"
        ) from e
    except subprocess.CalledProcessError as e:
        # ffmpeg/ffprobe ran but returned a non-zero exit code
        raise RuntimeError(
            f"Command failed with exit code {e.returncode}.\n"
            f"Command: {full_cmd}\n"
            f"Stdout:\n{e.stdout}\n\nStderr:\n{e.stderr}"
        ) from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_to_wav_16k_mono(
    input_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Convert any audio file to mono 16kHz WAV for Whisper.
    Returns the path to the converted .wav file.
    """
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / (input_path.stem + "_16k.wav")

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",        # mono
        "-ar",
        "16000",    # 16kHz
        str(out_path),
    ]
    _run_cmd(cmd)
    return out_path


def probe_audio(path: str | Path) -> tuple[float, int]:
    """
    Use ffprobe to get duration (seconds) and sample_rate.
    Returns (duration_seconds, sample_rate).
    """
    path = Path(path)

    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=sample_rate",
        "-of",
        "json",
        str(path),
    ]
    proc = _run_cmd(cmd)
    info = json.loads(proc.stdout)

    duration = float(info["format"]["duration"])

    # find first audio stream
    sr: int | None = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            sr = int(stream["sample_rate"])
            break

    if sr is None:
        # Fallback – should rarely happen if ffprobe is working,
        # but it's better than crashing.
        sr = 16000

    return duration, sr


def load_audio(input_path: str | Path) -> AudioFileInfo:
    """
    Main entry: convert & probe.
    Converts the input file to 16kHz mono WAV and probes duration + sample rate.
    """
    input_path = Path(input_path)
    wav_path = convert_to_wav_16k_mono(input_path)

    duration, sr = probe_audio(wav_path)

    return AudioFileInfo(
        path=str(wav_path),
        duration_seconds=duration,
        sample_rate=sr,
    )


def detect_silence_segments(audio_info: AudioFileInfo):
    """
    Optional VAD – left as future work.
    """
    # You can integrate webrtcvad or librosa here if you want.
    return []
