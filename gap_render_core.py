"""
Geekatplay Studio - Vladimir Chopine
Shared FastH3 render core for the StoryTeller suite.
Website: https://www.geekatplay.com

Model loading, sigma shift, sampling, dual VAE decode, segment checkpointing and
background-music mixing, shared by the batch renderer and the segment manager so
the two cannot drift apart.
"""

from fractions import Fraction
import hashlib
import json
import os
import re

import torch
import torchaudio

import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
from comfy_api.latest import InputImpl, Types
from comfy_api.latest._util import VideoCodec, VideoContainer
from comfy_extras import nodes_minimax_h3 as h3
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise

FPS = 24

MUSIC_MIX_MODES = [
    "blend with video audio",
    "ducking under video audio",
    "replace video audio",
]

_MODEL_CACHE = {}


def _get_cached_model(kind, key, loader_fn):
    entry = _MODEL_CACHE.get(kind)
    if entry is None or entry[0] != key:
        entry = (key, loader_fn())
        _MODEL_CACHE[kind] = entry
    return entry[1]


def _load_vae(vae_name):
    """Loads a VAE exactly the way comfy-core VAELoader does."""
    path = folder_paths.get_full_path_or_raise("vae", vae_name)
    sd = comfy.utils.load_torch_file(path)
    return comfy.sd.VAE(sd=sd)


def _apply_sigma_shift(model, shift_video, shift_audio):
    """Applies the MiniMax H3 video/audio flow shifts (ModelSamplingMiniMaxH3).

    FastH3 8-step will not converge on the default schedule, so this patch is
    mandatory, exactly as in the reference FastH3 workflow."""
    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    if hasattr(original, "noise_scale"):
        model_sampling.noise_scale = original.noise_scale
    m.add_object_patch("model_sampling", model_sampling)
    return m


def _segment_fingerprint(**parts):
    h = hashlib.blake2b(digest_size=8)
    h.update(json.dumps(parts, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _segment_path(prefix, index, fingerprint=None):
    """The saved file for one segment.

    The name is clean and numbered so the files are browsable and re-usable on their
    own - "KeyframeChain_segment_003.mp4", not a hash. The fingerprint that decides
    whether the segment can be resumed lives in a sidecar .json next to it, so
    changing a setting overwrites the stale segment instead of littering the output
    folder with orphans."""
    base = os.path.join(folder_paths.get_output_directory(), *prefix.replace("\\", "/").split("/"))
    os.makedirs(os.path.dirname(base) or base, exist_ok=True)
    return f"{base}_segment_{index:03d}.mp4"


def _sidecar_path(segment_path):
    return os.path.splitext(segment_path)[0] + ".json"


def _write_sidecar(path, fingerprint, info=None):
    data = {"fingerprint": fingerprint}
    if info:
        data.update(info)
    try:
        with open(_sidecar_path(path), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        print(f"[Geekatplay Studio] Warning: could not write sidecar for {os.path.basename(path)}: {exc}")


def _sidecar_matches(path, fingerprint):
    """True when the segment on disk was rendered with these exact settings."""
    side = _sidecar_path(path)
    if not os.path.exists(path) or not os.path.exists(side):
        return False
    try:
        with open(side, "r", encoding="utf-8") as f:
            return json.load(f).get("fingerprint") == fingerprint
    except Exception:
        return False


def _resume_segment(path, fingerprint, allowed=True):
    """Loads a finished segment when it is still valid and re-rendering was not forced."""
    if not allowed or not _sidecar_matches(path, fingerprint):
        return None
    return _load_segment(path)


def parse_rerender_list(text, total):
    """Reads '2, 5-7' into a set of 1-based segment numbers to force re-rendering."""
    forced = set()
    for chunk in re.split(r"[,\s]+", (text or "").strip()):
        if not chunk:
            continue
        if chunk.lower() in ("all", "*"):
            return set(range(1, total + 1))
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", chunk)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            forced.update(range(min(lo, hi), max(lo, hi) + 1))
        elif chunk.isdigit():
            forced.add(int(chunk))
        else:
            print(f"[Geekatplay Studio] Ignoring unreadable segment number '{chunk}'.")
    return {i for i in forced if 1 <= i <= total}


def _save_segment(path, frames, waveform, sample_rate):
    video = InputImpl.VideoFromComponents(Types.VideoComponents(
        images=frames,
        audio={"waveform": waveform, "sample_rate": sample_rate},
        frame_rate=Fraction(FPS)
    ))
    tmp = path + ".part"
    video.save_to(tmp, format=VideoContainer.MP4, codec=VideoCodec.H264)
    os.replace(tmp, path)


def _load_segment(path):
    try:
        components = InputImpl.VideoFromFile(path).get_components()
        if components.audio is None:
            return None
        return (components.images,
                components.audio["waveform"],
                components.audio["sample_rate"])
    except Exception as exc:
        print(f"[Geekatplay Studio] ignoring damaged checkpoint {os.path.basename(path)}: {exc}")
        return None


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


def _sample_segment(model, cond, latent, sampler_name, scheduler, steps, seed):
    noise = Noise_RandomNoise(seed)
    guider = Guider_Basic(model)
    guider.set_conds(cond)
    sampler = comfy.samplers.sampler_object(sampler_name)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, steps
    ).cpu()

    latent_image = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, {})
    samples = guider.sample(
        noise.generate_noise({"samples": latent_image}), latent_image, sampler, sigmas,
        denoise_mask=None, callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
    )
    return samples.to(comfy.model_management.intermediate_device())


def _decode_segment(samples, video_vae, audio_vae):
    video_latent = samples.unbind()[0] if samples.is_nested else samples
    images = video_vae.decode(video_latent)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    audio_latent = samples.unbind()[-1] if samples.is_nested else samples
    waveform = audio_vae.decode(audio_latent).movedim(-1, 1)
    std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    waveform /= std
    sample_rate = getattr(audio_vae, "audio_sample_rate_output",
                          getattr(audio_vae, "audio_sample_rate", 44100))
    return images, waveform, sample_rate


def _mix_background_music(base_wave, music_audio, num_frames, sample_rate, mode, music_gain, sfx_gain):
    target_samples = int(round(num_frames * sample_rate / FPS))
    base_wave = _fit_audio(base_wave, num_frames, sample_rate) * sfx_gain

    if not _is_valid_audio(music_audio):
        return base_wave

    try:
        m_wave = music_audio["waveform"].detach().clone().float()
        m_sr = music_audio.get("sample_rate", sample_rate) or sample_rate

        if m_wave.ndim == 2:
            m_wave = m_wave.unsqueeze(0)
        if m_wave.shape[-1] == 0:
            return base_wave

        if m_sr != sample_rate and m_wave.shape[-1] > 0:
            m_wave = torchaudio.functional.resample(m_wave, m_sr, sample_rate)

        base_ch = base_wave.shape[1] if base_wave.ndim == 3 else base_wave.shape[0]
        if m_wave.shape[1] < base_ch:
            m_wave = m_wave.repeat(1, base_ch, 1)
        elif m_wave.shape[1] > base_ch:
            m_wave = m_wave[:, :base_ch, :]

        m_len = m_wave.shape[-1]
        if m_len == 0:
            return base_wave

        if m_len < target_samples:
            repeats = (target_samples + m_len - 1) // m_len
            m_wave = m_wave.repeat(1, 1, repeats)[..., :target_samples]
        else:
            m_wave = m_wave[..., :target_samples]

        if mode in ("replace video audio", "replace audio"):
            return torch.clamp(m_wave * music_gain, -1.0, 1.0)

        if mode in ("ducking under video audio", "ducking under dialogue"):
            hop = max(1, sample_rate // 50)
            envelope = torch.abs(base_wave)
            kernel = torch.ones(1, 1, hop, device=base_wave.device) / hop
            if envelope.ndim == 3:
                envelope = envelope.mean(dim=1, keepdim=True)
            smoothed = torch.nn.functional.conv1d(envelope, kernel, padding=hop // 2)[..., :target_samples]
            duck_scale = torch.clamp(1.0 - (smoothed / 0.15), 0.15, 1.0)
            ducked_music = m_wave * (music_gain * duck_scale)
            return torch.clamp(base_wave + ducked_music, -1.0, 1.0)

        return torch.clamp(base_wave + (m_wave * music_gain), -1.0, 1.0)
    except Exception as exc:
        print(f"[Geekatplay Studio] Warning: failed mixing background audio ({exc}), continuing with native audio.")
        return base_wave
