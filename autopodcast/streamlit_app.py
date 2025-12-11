# autopodcast/audio_preprocessing.py

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

from models import AudioFileInfo  # Adjust the import path to match your project structure
from shutil import which 
# IMPORTANT: update these two paths if your ffmpeg build is in a different folder
FFMPEG_PATH = which("ffmpeg")
FFPROBE_PATH = which("ffprobe")


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    """
    Run a subprocess command and raise if it fails.
    We capture stdout/stderr so we can inspect them on error.
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return proc
    except FileNotFoundError as e:
        # This is exactly the error you're seeing
        raise FileNotFoundError(
            f"Failed to run external command. "
            f"Command: {cmd}\n"
            f"Make sure the path to ffmpeg/ffprobe is correct.\n"
            f"Original error: {e}"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed with exit code {e.returncode}.\n"
            f"Command: {cmd}\n"
            f"Stdout:\n{e.stdout}\n\nStderr:\n{e.stderr}"
        ) from e


def convert_to_wav_16k_mono(input_path: str) -> str:
    """
    Convert any input audio file to 16kHz mono WAV using ffmpeg.
    Returns the path to the converted wav file.
    """
    in_path = Path(input_path)
    out_path = in_path.with_suffix(".wav")

    cmd = [
        FFMPEG_PATH,            # <--- absolute path to ffmpeg.exe
        "-y",                   # overwrite output
        "-i", str(in_path),
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz
        str(out_path),
    ]

    _run_cmd(cmd)
    return str(out_path)


def probe_audio(path: str) -> Tuple[float, int]:
    """
    Use ffprobe to get (duration_seconds, sample_rate) for a WAV file.
    """
    audio_path = Path(path)

    cmd = [
        FFPROBE_PATH,           # <--- absolute path to ffprobe.exe
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    proc = _run_cmd(cmd)

    # ffprobe with the options above prints two lines:
    #   duration
    #   sample_rate
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(
            f"Unexpected ffprobe output for file {audio_path}:\n{proc.stdout}"
        )

    duration_str, sr_str = lines[0], lines[1]
    duration = float(duration_str)
    sample_rate = int(sr_str)

    return duration, sample_rate


def load_audio(input_path: str) -> AudioFileInfo:
    """
    High-level helper:
    - Convert input file to WAV 16k mono.
    - Probe duration and sample rate.
    - Return an AudioFileInfo object.
    """
    input_path = str(input_path)
    wav_path = convert_to_wav_16k_mono(input_path)
    duration, sr = probe_audio(wav_path)

    info = AudioFileInfo(
        path=wav_path,
        original_path=input_path,
        duration_seconds=duration,
        sample_rate=sr,
    )
    return info
