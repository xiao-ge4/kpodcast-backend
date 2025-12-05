import os
import io
from typing import Optional, List
from pydub import AudioSegment
from pydub.utils import which


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _ensure_ffmpeg() -> Optional[str]:
    """尽力定位并注入 ffmpeg/ffprobe 到当前进程环境，返回 ffmpeg 可执行路径。"""
    # 1) PATH/环境变量
    p = os.environ.get("FFMPEG_BINARY") or which("ffmpeg")
    if p and os.path.exists(p):
        AudioSegment.converter = p
        AudioSegment.ffmpeg = p
        # 尝试 ffprobe
        probedir = os.path.dirname(p)
        probe = os.path.join(probedir, "ffprobe.exe") if os.name == 'nt' else os.path.join(probedir, "ffprobe")
        if os.path.exists(probe):
            AudioSegment.ffprobe = probe
            os.environ["FFPROBE"] = probe
        if probedir not in (os.environ.get("PATH") or ""):
            os.environ["PATH"] = probedir + os.pathsep + os.environ.get("PATH", "")
        return p
    # 2) 常见安装位置
    candidates = [
        r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        r"C:\\Program Files\\FFmpeg\\bin\\ffmpeg.exe",
        r"C:\\ffmpeg\\bin\\ffmpeg.exe",
        r"C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe",
    ]
    # 3) WinGet 目录下递归查找
    local_pkg = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(local_pkg):
        for root, _, files in os.walk(local_pkg):
            if "ffmpeg.exe" in files:
                candidates.append(os.path.join(root, "ffmpeg.exe"))
                break
    for c in candidates:
        if c and os.path.exists(c):
            os.environ["FFMPEG_BINARY"] = c
            AudioSegment.converter = c
            AudioSegment.ffmpeg = c
            dirc = os.path.dirname(c)
            probe = os.path.join(dirc, "ffprobe.exe") if os.name == 'nt' else os.path.join(dirc, "ffprobe")
            if os.path.exists(probe):
                AudioSegment.ffprobe = probe
                os.environ["FFPROBE"] = probe
            if dirc not in (os.environ.get("PATH") or ""):
                os.environ["PATH"] = dirc + os.pathsep + os.environ.get("PATH", "")
            return c
    return None


def mix_intro_with_voice(intro_path: Optional[str], voice_path: str, out_path: str, duck_db: float = -6.0) -> str:
    """将片头音乐与语音文件做淡入/淡出拼接并导出 mp3。"""
    _ensure_ffmpeg()
    voice = AudioSegment.from_file(voice_path)
    if intro_path and os.path.exists(intro_path):
        intro = AudioSegment.from_file(intro_path)
        intro = intro.fade_in(100).fade_out(400) + duck_db
        mixed = intro.append(voice, crossfade=200)
    else:
        mixed = voice
    mixed.export(out_path, format="mp3", bitrate="192k")
    return out_path


def concat_voice_segments(audio_bytes_list: List[bytes], pause_ms: int = 200) -> AudioSegment:
    """把多段 mp3 二进制拼接为一个 AudioSegment，中间加入短暂停顿。"""
    _ensure_ffmpeg()
    final = AudioSegment.silent(duration=100)
    gap = AudioSegment.silent(duration=max(0, int(pause_ms)))
    for b in audio_bytes_list:
        seg = AudioSegment.from_file(io.BytesIO(b), format="mp3")
        final = final.append(seg, crossfade=50).append(gap, crossfade=0)
    return final


def export_with_intro(audio_segment: AudioSegment, out_path: str, intro_path: Optional[str] = None) -> str:
    """可选在音频前添加片头，再导出 mp3。"""
    _ensure_ffmpeg()
    if intro_path and os.path.exists(intro_path):
        intro = AudioSegment.from_file(intro_path).fade_in(100).fade_out(400) - 6
        audio_segment = intro.append(audio_segment, crossfade=200)
    audio_segment.export(out_path, format="mp3", bitrate="192k")
    return out_path

