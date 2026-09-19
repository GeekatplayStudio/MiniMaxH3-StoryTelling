"""
Geekatplay Studio - Vladimir Chopine
StoryTeller Batch Renderer for MiniMax FastH3 Lite & ComfyUI
Website: https://www.geekatplay.com

Renders every segment of a GAP_STORY_PLAN sequentially with FastH3 8-step, with
resume capability, seam trimming, and fail-safe background music mixing.
"""

from fractions import Fraction
import os

import torch
import torchaudio

import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, Types
from comfy_extras import nodes_minimax_h3 as h3

from .story_director import free_ollama_vram as _free_ollama_vram
from .gap_render_core import (
    FPS,
    MUSIC_MIX_MODES,
    _apply_sigma_shift,
    _decode_segment,
    _fit_audio,
    _get_cached_model,
    _load_vae,
    _mix_background_music,
    _sample_segment,
    _resume_segment,
    _save_segment,
    _segment_fingerprint,
    _segment_path,
    _write_sidecar,
    parse_rerender_list,
)


class GAPStoryTellerBatchRenderer:
    """Geekatplay Studio - Vladimir Chopine.
    StoryTeller Batch Segment Renderer.
    Consumes a GAP_STORY_PLAN and renders all story segments sequentially
    using FastVideo FastH3 8-step (or base MiniMax H3).
    Supports checkpointing, resume, seamless audio/video stitching, and
    optional user background music mixing with ducking."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_plan": ("GAP_STORY_PLAN",),
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "default": "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
                    "tooltip": "FastVideo FastH3 8-step checkpoint or MiniMax H3 model."
                }),
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {
                    "default": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    "tooltip": "Qwen3-VL text encoder for MiniMax H3."
                }),
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "default": "minimax_h3_video_vae_fp16.safetensors",
                    "tooltip": "MiniMax H3 video VAE."
                }),
                "audio_vae_name": (folder_paths.get_filename_list("vae"), {
                    "default": "minimax_h3_audio_vae_fp32.safetensors",
                    "tooltip": "MiniMax H3 audio VAE for native synchronized audio."
                }),
                "steps": ("INT", {
                    "default": 8, "min": 1, "max": 100, "step": 1,
                    "tooltip": "FastVideo FastH3 8-step uses 8 steps. Standard MiniMax H3 uses 20."
                }),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "shift_video": ("FLOAT", {
                    "default": 10.0, "min": 0.01, "max": 100.0, "step": 0.01,
                    "tooltip": "MiniMax H3 video flow shift (ModelSamplingMiniMaxH3). 10.0 matches the reference FastH3 workflow."
                }),
                "shift_audio": ("FLOAT", {
                    "default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01,
                    "tooltip": "MiniMax H3 audio flow shift. Keep at 3.0 unless the soundtrack drifts."
                }),
                "noise_seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Base seed. Segment i uses noise_seed + i."
                }),
            },
            "optional": {
                "background_music": ("AUDIO", {
                    "tooltip": "Optional user-supplied background music to blend into the final video."
                }),
                "music_mix_mode": (MUSIC_MIX_MODES, {
                    "default": "blend with video audio",
                    "tooltip": "How user background music interacts with the native AI soundscape & dialogue."
                }),
                "music_gain": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                "sfx_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "save_segments": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save each rendered segment to disk immediately so progress is never lost."
                }),
                "resume": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse finished segments already on disk when parameters are unchanged."
                }),
                "segment_prefix": ("STRING", {
                    "default": "storyteller/MiniMaxH3",
                    "tooltip": "Where individual segments are written inside the ComfyUI output "
                               "directory, as <prefix>_segment_001.mp4 and so on."
                }),
                "free_ollama_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload any model Ollama is holding before the video models load. "
                               "On a single card the Story Director's LLM would otherwise still "
                               "be resident while MiniMax H3 needs the same memory."
                }),
                "rerender_segments": ("STRING", {
                    "default": "",
                    "tooltip": "Segment numbers to force a re-render of, e.g. '3' or '2, 5-7'. "
                               "Everything else is reused from disk. Empty re-renders nothing."
                }),
                "trim_seam_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Drop the duplicate first frame (and its audio samples) of every segment after the first."
                }),
                "low_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Store accumulated frames in 8-bit RAM to conserve memory on longer videos."
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "frames", "audio", "segment_paths", "segment_prompts")
    FUNCTION = "render_story"
    CATEGORY = "Geekatplay/StoryTeller"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def render_story(self, story_plan, unet_name, clip_name, vae_name, audio_vae_name,
                     steps=8, sampler_name="euler", scheduler="simple",
                     shift_video=10.0, shift_audio=3.0, noise_seed=42,
                     background_music=None, music_mix_mode="blend with video audio",
                     music_gain=0.35, sfx_gain=1.0, save_segments=True, resume=True,
                     segment_prefix="storyteller/MiniMaxH3", free_ollama_vram=True,
                     rerender_segments="", trim_seam_frame=True, low_memory=True, **kwargs):

        if not story_plan or "segments" not in story_plan or not story_plan["segments"]:
            raise ValueError("Story plan has no segments to render.")

        if free_ollama_vram:
            _free_ollama_vram()

        segments = story_plan["segments"]
        width = story_plan.get("width", 1344)
        height = story_plan.get("height", 768)

        print(f"[Geekatplay Studio] Preparing to render {len(segments)} segments at {width}x{height}...")

        model = _get_cached_model("model", unet_name,
                                  lambda: comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", unet_name)))
        clip = _get_cached_model("clip", clip_name,
                                 lambda: comfy.sd.load_clip(ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", clip_name)],
                                                            embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                                            clip_type=comfy.sd.CLIPType.MINIMAX))
        video_vae = _get_cached_model("vae", vae_name, lambda: _load_vae(vae_name))
        audio_vae = _get_cached_model("audio_vae", audio_vae_name, lambda: _load_vae(audio_vae_name))

        model = _apply_sigma_shift(model, shift_video, shift_audio)

        all_frames = []
        all_waves = []
        segment_files = []
        prompt_log = []
        sample_rate = None

        rerender_set = parse_rerender_list(rerender_segments, len(segments))
        if rerender_set:
            print(f"[Geekatplay Studio] Forcing a re-render of segment(s): "
                  f"{', '.join(str(i) for i in sorted(rerender_set))}")

        pbar = comfy.utils.ProgressBar(len(segments))

        for i, seg in enumerate(segments):
            seg_idx = seg.get("segment_index", i + 1)
            prompt = seg.get("prompt", "")
            duration = seg.get("duration", 5.0)
            frame_count = seg.get("frame_count", 124)
            seed = noise_seed + i

            fingerprint = _segment_fingerprint(
                unet=unet_name, clip=clip_name, vae=vae_name, audio_vae=audio_vae_name,
                prompt=prompt, width=width, height=height, frames=frame_count,
                steps=steps, sampler=sampler_name, scheduler=scheduler, seed=seed,
                shift_video=shift_video, shift_audio=shift_audio
            )
            prompt_log.append(
                f"{'=' * 72}\nSEGMENT {seg_idx}/{len(segments)}\n{'=' * 72}\n"
                f"canvas      : {width}x{height}   frames: {frame_count} ({duration}s)   "
                f"seed: {seed}\n"
                f"first/last  : none - this is text-to-video, the frames are generated\n"
                f"{'-' * 72}\n{prompt}\n")

            path = _segment_path(segment_prefix, seg_idx)
            forced = seg_idx in rerender_set

            restored = _resume_segment(path, fingerprint, allowed=resume and not forced)
            if restored is not None:
                frames, wave, sr = restored
                print(f"[Geekatplay Studio] Reusing saved segment {seg_idx}/{len(segments)}: {os.path.basename(path)}")
            else:
                print(f"[Geekatplay Studio] Rendering segment {seg_idx}/{len(segments)} ({frame_count} frames, seed {seed})...")
                tokens = clip.tokenize(prompt, images=[])
                cond = clip.encode_from_tokens_scheduled(tokens)
                latent, _ = h3._empty_av_latent(width, height, frame_count)

                samples = _sample_segment(model, cond, latent, sampler_name, scheduler, steps, seed)
                frames, wave, sr = _decode_segment(samples, video_vae, audio_vae)

                del cond, latent, samples
                comfy.model_management.soft_empty_cache()

                if save_segments:
                    _save_segment(path, frames, wave, sr)
                    _write_sidecar(path, fingerprint, {
                        "segment": seg_idx, "duration": duration, "frames": frame_count,
                        "seed": seed, "prompt": prompt})
                    print(f"[Geekatplay Studio] Saved {os.path.basename(path)}")

            if save_segments:
                segment_files.append(path)
            sample_rate = sample_rate or sr

            if i > 0 and trim_seam_frame and frames.shape[0] > 1:
                frames = frames[1:]
                wave = wave[..., int(round(sr / FPS)):]

            if sr != sample_rate:
                wave = torchaudio.functional.resample(wave, sr, sample_rate)

            wave = _fit_audio(wave, frames.shape[0], sample_rate)

            if low_memory:
                frames = (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

            all_frames.append(frames)
            all_waves.append(wave)
            pbar.update(1)

        final_frames = torch.cat(all_frames, dim=0)
        all_frames.clear()
        if final_frames.dtype == torch.uint8:
            final_frames = final_frames.float().div_(255.0)

        min_channels = min(w.shape[1] for w in all_waves)
        base_waveform = torch.cat([w[:, :min_channels] for w in all_waves], dim=-1)
        all_waves.clear()

        final_waveform = _mix_background_music(
            base_wave=base_waveform,
            music_audio=background_music,
            num_frames=final_frames.shape[0],
            sample_rate=sample_rate,
            mode=music_mix_mode,
            music_gain=music_gain,
            sfx_gain=sfx_gain
        )

        audio_output = {
            "waveform": final_waveform,
            "sample_rate": sample_rate,
        }

        final_video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=final_frames,
            audio=audio_output,  # type: ignore
            frame_rate=Fraction(FPS)
        ))

        paths_str = "\n".join(segment_files)
        print(f"[Geekatplay Studio] Finished assembling movie: {final_frames.shape[0]} frames ({final_frames.shape[0]/FPS:.1f}s).")
        return (final_video, final_frames, audio_output, paths_str, "\n".join(prompt_log))
