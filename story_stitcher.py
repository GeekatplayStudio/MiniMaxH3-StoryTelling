"""
Geekatplay Studio - Vladimir Chopine
StoryVideoStitcher for MiniMax FastH3 Lite & ComfyUI
Website: https://www.geekatplay.com

Stitches multiple video segments into one continuous video, with optional
user background music mixing, dialogue ducking, gain controls, and seam trimming.
"""

from fractions import Fraction
import os

import torch
import torchaudio

from comfy_api.latest import InputImpl, Types
from comfy_api.latest._util import VideoCodec, VideoContainer

FPS = 24


def _is_valid_audio(audio):
    """Validates that audio is a valid dictionary with non-empty waveform tensor."""
    if audio is None or not isinstance(audio, dict):
        return False
    wave = audio.get("waveform")
    if wave is None or not isinstance(wave, torch.Tensor):
        return False
    if wave.numel() == 0 or wave.shape[-1] == 0:
        return False
    return True


def _fit_audio(wave, num_frames, sample_rate):
    target = int(round(num_frames * sample_rate / FPS))
    length = wave.shape[-1]
    if length > target:
        return wave[..., :target]
    if length < target:
        pad = torch.zeros(*wave.shape[:-1], target - length, dtype=wave.dtype, device=wave.device)
        return torch.cat([wave, pad], dim=-1)
    return wave


class GAPStoryVideoStitcher:
    """Geekatplay Studio - Vladimir Chopine.
    Stitches multiple video segments into one seamless video, with optional
    background music mixing, gain adjustments, and seam trimming."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "music_mix_mode": (["blend with video audio", "ducking under video audio", "replace video audio"], {
                    "default": "blend with video audio"
                }),
                "music_gain": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                "video_audio_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "trim_seam_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Drop the first frame of subsequent segments to avoid duplicate boundary frames."
                }),
            },
            "optional": {
                "video_1": ("VIDEO",),
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "video_4": ("VIDEO",),
                "video_5": ("VIDEO",),
                "segment_paths": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional newline-separated list of mp4 segment paths to load and stitch."
                }),
                "background_music": ("AUDIO", {
                    "tooltip": "Optional background music bed."
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "FLOAT", "INT")
    RETURN_NAMES = ("video", "frames", "audio", "duration_seconds", "frame_count")
    FUNCTION = "stitch_videos"
    CATEGORY = "Geekatplay/StoryTeller"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def stitch_videos(self, music_mix_mode="blend with video audio", music_gain=0.35,
                      video_audio_gain=1.0, trim_seam_frame=True,
                      video_1=None, video_2=None, video_3=None, video_4=None, video_5=None,
                      segment_paths="", background_music=None, **kwargs):

        video_inputs = [v for v in (video_1, video_2, video_3, video_4, video_5) if v is not None]

        if segment_paths and segment_paths.strip():
            for line in segment_paths.strip().splitlines():
                path = line.strip()
                if path and os.path.exists(path):
                    try:
                        v = InputImpl.VideoFromFile(path)
                        video_inputs.append(v)
                    except Exception as e:
                        print(f"[Geekatplay Studio] Failed to load {path}: {e}")

        if not video_inputs:
            raise ValueError("GAPStoryVideoStitcher: Connect at least one video socket or provide valid segment_paths.")

        all_frames = []
        all_waves = []
        target_sample_rate = 44100

        for i, vid in enumerate(video_inputs):
            comp = vid.get_components()
            frames = comp.images
            audio = comp.audio

            if audio is not None and "waveform" in audio:
                wave = audio["waveform"].detach().float()
                sr = audio.get("sample_rate", target_sample_rate)
            else:
                sr = target_sample_rate
                wave = torch.zeros(1, 2, int(round(frames.shape[0] * sr / FPS)))

            if i == 0:
                target_sample_rate = sr
            elif sr != target_sample_rate:
                wave = torchaudio.functional.resample(wave, sr, target_sample_rate)

            if i > 0 and trim_seam_frame and frames.shape[0] > 1:
                frames = frames[1:]
                wave = wave[..., int(round(target_sample_rate / FPS)):]

            wave = _fit_audio(wave, frames.shape[0], target_sample_rate)
            all_frames.append(frames)
            all_waves.append(wave)

        stitched_frames = torch.cat(all_frames, dim=0)

        min_channels = min(w.shape[1] for w in all_waves)
        stitched_wave = torch.cat([w[:, :min_channels] for w in all_waves], dim=-1) * video_audio_gain
        stitched_wave = _fit_audio(stitched_wave, stitched_frames.shape[0], target_sample_rate)

        if background_music is not None and _is_valid_audio(background_music):
            try:
                m_wave = background_music["waveform"].detach().clone().float()
                m_sr = background_music.get("sample_rate", target_sample_rate) or target_sample_rate
                if m_wave.ndim == 2:
                    m_wave = m_wave.unsqueeze(0)
                if m_sr != target_sample_rate and m_wave.shape[-1] > 0:
                    m_wave = torchaudio.functional.resample(m_wave, m_sr, target_sample_rate)

                if m_wave.shape[1] < min_channels:
                    m_wave = m_wave.repeat(1, min_channels, 1)
                elif m_wave.shape[1] > min_channels:
                    m_wave = m_wave[:, :min_channels, :]

                total_samples = stitched_wave.shape[-1]
                if m_wave.shape[-1] > 0:
                    if m_wave.shape[-1] < total_samples:
                        reps = (total_samples + m_wave.shape[-1] - 1) // m_wave.shape[-1]
                        m_wave = m_wave.repeat(1, 1, reps)[..., :total_samples]
                    else:
                        m_wave = m_wave[..., :total_samples]

                    if music_mix_mode == "replace video audio":
                        stitched_wave = torch.clamp(m_wave * music_gain, -1.0, 1.0)
                    elif music_mix_mode == "ducking under video audio":
                        hop = max(1, target_sample_rate // 50)
                        env = torch.abs(stitched_wave)
                        kernel = torch.ones(1, 1, hop, device=stitched_wave.device) / hop
                        smoothed = torch.nn.functional.conv1d(env.mean(dim=1, keepdim=True), kernel, padding=hop // 2)[..., :total_samples]
                        duck_scale = torch.clamp(1.0 - (smoothed / 0.15), 0.15, 1.0)
                        stitched_wave = torch.clamp(stitched_wave + (m_wave * music_gain * duck_scale), -1.0, 1.0)
                    else:
                        stitched_wave = torch.clamp(stitched_wave + (m_wave * music_gain), -1.0, 1.0)
            except Exception as exc:
                print(f"[Geekatplay Studio] Warning: failed mixing background audio in stitcher ({exc}), continuing with native audio.")

        final_audio = {
            "waveform": stitched_wave,
            "sample_rate": target_sample_rate,
        }

        final_video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=stitched_frames,
            audio=final_audio,  # type: ignore
            frame_rate=Fraction(FPS)
        ))

        duration_sec = float(stitched_frames.shape[0] / FPS)
        return (final_video, stitched_frames, final_audio, duration_sec, stitched_frames.shape[0])
