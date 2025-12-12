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


def mix_intro_voice_with_bgm(intro_voice: AudioSegment, bgm_path: str, out_path: str, 
                              bgm_volume_db: float = -8.0, fade_out_ms: int = 500) -> str:
    """
    将片头语音与背景音乐混合
    
    Args:
        intro_voice: 片头语音 AudioSegment
        bgm_path: 背景音乐文件路径
        out_path: 输出文件路径
        bgm_volume_db: 背景音乐音量调整（相对于原音量的dB值）
        fade_out_ms: 背景音乐淡出时长（毫秒）
    
    Returns:
        输出文件路径
    """
    _ensure_ffmpeg()
    
    if not os.path.exists(bgm_path):
        # 如果没有背景音乐，直接导出语音
        intro_voice.export(out_path, format="mp3", bitrate="192k")
        return out_path
    
    # 加载背景音乐
    bgm = AudioSegment.from_file(bgm_path)
    
    # 计算需要的背景音乐长度（语音长度 + 额外的淡出时间）
    target_length = len(intro_voice) + fade_out_ms + 200
    
    # 如果背景音乐太短，循环播放
    if len(bgm) < target_length:
        loops_needed = (target_length // len(bgm)) + 1
        bgm = bgm * loops_needed
    
    # 截取需要的长度
    bgm = bgm[:target_length]
    
    # 调整背景音乐音量并添加淡入淡出
    bgm = bgm + bgm_volume_db
    bgm = bgm.fade_in(300).fade_out(fade_out_ms)
    
    # 在语音前添加一小段纯背景音乐（让背景音乐先起来）
    lead_in_ms = 500
    
    # 混合：背景音乐从头开始，语音在 lead_in_ms 后开始
    # 先创建一个与背景音乐等长的静音轨道
    voice_track = AudioSegment.silent(duration=lead_in_ms) + intro_voice
    # 确保语音轨道与背景音乐等长
    if len(voice_track) < len(bgm):
        voice_track = voice_track + AudioSegment.silent(duration=len(bgm) - len(voice_track))
    elif len(voice_track) > len(bgm):
        bgm = bgm + AudioSegment.silent(duration=len(voice_track) - len(bgm))
    
    # 叠加混合
    mixed = bgm.overlay(voice_track)
    
    # 导出
    mixed.export(out_path, format="mp3", bitrate="192k")
    return out_path


def _adjust_bgm_length_stretch(bgm: AudioSegment, target_length: int) -> AudioSegment:
    """
    使用变速方式调整BGM长度
    
    Args:
        bgm: 原始BGM音频
        target_length: 目标长度（毫秒）
    
    Returns:
        调整后的BGM音频
    """
    if len(bgm) == target_length:
        return bgm
    
    # 计算变速比例
    speed_ratio = len(bgm) / target_length
    
    # 使用 pydub 的变速功能
    # 变速会同时改变音调，这里通过改变采样率来实现
    # speed_ratio > 1 表示需要加速（BGM比目标长）
    # speed_ratio < 1 表示需要减速（BGM比目标短）
    
    # 通过改变帧率来变速
    new_frame_rate = int(bgm.frame_rate * speed_ratio)
    stretched = bgm._spawn(bgm.raw_data, overrides={'frame_rate': new_frame_rate})
    # 转换回原始帧率以保持兼容性
    stretched = stretched.set_frame_rate(bgm.frame_rate)
    
    return stretched


def _adjust_bgm_length_loop(bgm: AudioSegment, target_length: int, crossfade_ms: int = 150) -> AudioSegment:
    """
    使用循环/裁剪方式调整BGM长度
    
    Args:
        bgm: 原始BGM音频
        target_length: 目标长度（毫秒）
        crossfade_ms: 循环接缝处的交叉淡化时长（毫秒）
    
    Returns:
        调整后的BGM音频
    """
    if len(bgm) >= target_length:
        # BGM足够长，直接裁剪
        return bgm[:target_length]
    
    # BGM太短，需要循环
    result = bgm
    
    while len(result) < target_length:
        # 计算还需要多少长度
        remaining = target_length - len(result)
        
        if remaining <= crossfade_ms:
            # 剩余长度太短，不足以做交叉淡化，直接用静音填充
            result = result + AudioSegment.silent(duration=remaining)
            break
        
        # 取需要的长度（或整个BGM）
        chunk_length = min(len(bgm), remaining)
        chunk = bgm[:chunk_length]
        
        # 使用交叉淡化拼接
        if len(result) > crossfade_ms and len(chunk) > crossfade_ms:
            result = result.append(chunk, crossfade=crossfade_ms)
        else:
            result = result + chunk
    
    # 确保精确长度
    return result[:target_length]


def export_with_dynamic_intro(main_audio: AudioSegment, intro_voice: Optional[AudioSegment], 
                               bgm_path: Optional[str], out_path: str,
                               bgm_strategy: str = "loop", loop_crossfade_ms: int = 150) -> str:
    """
    使用动态生成的片头语音和背景音乐，与主音频拼接
    
    Args:
        main_audio: 主音频内容
        intro_voice: 片头语音（如果为None则只用背景音乐）
        bgm_path: 背景音乐路径
        out_path: 输出文件路径
        bgm_strategy: BGM长度调整策略 ("stretch" 或 "loop")
        loop_crossfade_ms: 循环策略的交叉淡化时长（毫秒）
    
    Returns:
        输出文件路径
    """
    _ensure_ffmpeg()
    
    if intro_voice is None and (bgm_path is None or not os.path.exists(bgm_path)):
        # 没有片头，直接导出主音频
        main_audio.export(out_path, format="mp3", bitrate="192k")
        return out_path
    
    if intro_voice is None:
        # 只有背景音乐，没有语音（通用风格）
        bgm = AudioSegment.from_file(bgm_path)
        # 截取合适长度（比如5秒）
        intro_length = min(5000, len(bgm))
        intro = bgm[:intro_length].fade_in(100).fade_out(400) - 6
        final_audio = intro.append(main_audio, crossfade=200)
    else:
        # 有片头语音，需要与背景音乐混合
        if bgm_path and os.path.exists(bgm_path):
            # 混合片头语音和背景音乐
            bgm = AudioSegment.from_file(bgm_path)
            
            # 计算需要的背景音乐长度
            lead_in_ms = 500  # 语音前的纯背景音乐时长
            fade_out_ms = 500  # 淡出时长
            target_length = lead_in_ms + len(intro_voice) + fade_out_ms
            
            # 根据策略调整BGM长度
            if bgm_strategy == "stretch":
                bgm = _adjust_bgm_length_stretch(bgm, target_length)
            else:  # "loop" 或其他
                bgm = _adjust_bgm_length_loop(bgm, target_length, crossfade_ms=loop_crossfade_ms)
            
            bgm = bgm - 8  # 降低背景音乐音量
            bgm = bgm.fade_in(300).fade_out(fade_out_ms)
            
            # 在语音前添加一小段纯背景音乐
            voice_track = AudioSegment.silent(duration=lead_in_ms) + intro_voice
            
            # 确保两轨道等长
            if len(voice_track) < len(bgm):
                voice_track = voice_track + AudioSegment.silent(duration=len(bgm) - len(voice_track))
            elif len(voice_track) > len(bgm):
                bgm = bgm + AudioSegment.silent(duration=len(voice_track) - len(bgm))
            
            intro = bgm.overlay(voice_track)
        else:
            # 没有背景音乐，只用语音
            intro = intro_voice
        
        # 拼接片头和主音频
        final_audio = intro.append(main_audio, crossfade=200)
    
    final_audio.export(out_path, format="mp3", bitrate="192k")
    return out_path

