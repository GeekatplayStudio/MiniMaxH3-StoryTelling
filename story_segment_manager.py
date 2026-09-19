"""
Geekatplay Studio - Vladimir Chopine
Segment Manager for the StoryTeller Keyframe Chain
Website: https://www.geekatplay.com

Accepts as many segments as you connect and renders them as one continuous movie.

The keyframe chain is the point of this node. With keyframes A, B, C, D it renders:

    segment 1:  first = A, last = B
    segment 2:  first = B, last = C
    segment 3:  first = C, last = D
    segment 4:  first = D, no last frame   (the tail, optional)

Each image is therefore the closing frame of one segment and the opening frame of
the next, which is what keeps the cuts from jumping.
"""

from fractions import Fraction
import os
import re

import torch
import torchaudio

import comfy.model_management
import comfy.sd
import comfy.samplers
import comfy.utils
import folder_paths
import node_helpers
from comfy_api.latest import InputImpl, Types, io
from comfy_extras import nodes_minimax_h3 as h3

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
from .story_director import (
    ASPECT_RATIOS,
    free_ollama_vram as _free_ollama_vram,
    PACING_MOODS,
    VISUAL_STYLES,
    _style_clause,
    _format_timecode,
    align_frame_count,
    FRAME_SIZES,
    resolve_canvas,
)

MAX_SEGMENTS = 12

# What kind of film this is, as a clause appended next to the style lock.
VIDEO_TYPES = {
    "Narrative Scene": "a narrative scene shot for continuity, cuts motivated by the action",
    "Music Video": "a music video, staging cut to the beat, performance-forward framing",
    "Commercial / Product": "a commercial spot, the product held clean and hero-lit in frame",
    "Documentary / Observational": "observational documentary coverage, unforced framing, available light",
    "Fashion Film": "a fashion film, silhouette and fabric movement prioritised over plot",
    "Title Sequence": "a title sequence, graphic composition and rhythm over narrative",
    "Trailer / Teaser": "a trailer cut, escalating shot lengths and withheld reveals",
    "Nature / Landscape": "landscape cinematography, the environment as the subject",
}

ASPECT_CHOICES = ["match first keyframe"] + list(ASPECT_RATIOS.keys())

DURATION_SOURCES = [
    "from segment generator",
    "manager override (all segments)",
    "clamp generator to manager value",
]

# "at 0.00 seconds" contains a full stop, so the preamble cannot be matched by
# sentence punctuation. It is bounded by its closing phrase instead.
# How hard to push a style on top of the supplied frames. In image-to-video the frames
# already carry the look, so overriding it is usually wrong.
STYLE_MODES = [
    "none - send only my prompt (recommended)",
    "ask the model to follow the supplied frames",
    "light style note",
    "full style lock",
]

# The "arrive on the final frame" sentence the generator adds for paired segments.
_CLOSING_CLAIM = re.compile(
    r"\s*The motion ends on the composition of the final frame[^.]*\.\s*", re.IGNORECASE)

_ANCHOR_END = "is fully referenced."
_ANCHOR_SCAN = 400  # the preamble never runs longer than this


def _ordered(autogrow_dict):
    """Autogrow hands back {name: value}; sort by the trailing index, drop the gaps."""
    if not autogrow_dict:
        return []
    def idx(name):
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else 0
    return [v for _k, v in sorted(autogrow_dict.items(), key=lambda kv: idx(kv[0])) if v is not None]


def _style_note(mode, visual_style, pacing_and_mood, video_type):
    """The clause appended to every segment, sized to how much it should override."""
    if mode == "none - send only my prompt (recommended)":
        return ""
    if mode == "ask the model to follow the supplied frames":
        return ("Keep the look, colour, lighting and content of the supplied frames. "
                "Do not restyle, do not add anything that is not already in frame.")
    if mode == "light style note":
        return (f"Style: {visual_style}. {VIDEO_TYPES.get(video_type, '')}. "
                "Otherwise keep the look of the supplied frames.")
    return f"{_style_clause(visual_style, pacing_and_mood)} This is {VIDEO_TYPES.get(video_type, '')}."


def _rebuild_anchor(shot_text, has_last_frame, duration, wanted=False):
    """Normalises the <Picture N> preamble, which is off unless explicitly asked for.

    <Picture N> is reference-to-video vocabulary. First and last frames are anchored as
    keyframes and need no preamble at all, and the "<Picture 2> at the end" sentence was
    an extrapolation of mine that does not appear in the official workflow. Both are
    stripped here; only the one real sentence is re-added, and only on request."""
    body = (shot_text or "").strip()
    # The phrase only ever appears in the preamble, and the preamble is at most two
    # sentences, so the last occurrence near the front marks where the shot begins.
    # ("[Shot 1]" cannot be used as the boundary: the preamble itself contains it.)
    cut = body.rfind(_ANCHOR_END, 0, _ANCHOR_SCAN)
    if cut != -1:
        body = body[cut + len(_ANCHOR_END):].strip()
    # The generator writes every segment as if a closing frame follows, because only the
    # manager knows which one is the tail. Strip that promise from the tail.
    if not has_last_frame:
        body = _CLOSING_CLAIM.sub("", body).strip()
        body = re.sub(r"\s{2,}", " ", body)

    if not wanted:
        return body
    anchor = ("For the target video, at 0.00 seconds into the target video, "
              "<Picture 1> (from [Shot 1]) is fully referenced.")
    return f"{anchor} {body}".strip()


def _build_conditioning(clip, video_vae, prompt, first_frame, last_frame, width, height, length):
    """MiniMaxH3ImageToVideo, inlined so the manager can drive it per segment."""
    latent, frame_count = h3._empty_av_latent(width, height, length)

    images = []
    keyframes = []
    if first_frame is not None:
        img = h3._resize(first_frame[:1], width, height, "disabled")
        images.append(img)
        keyframes.append({"resolved_frame_index": 0, "image": img})
    if last_frame is not None:
        img = h3._resize(last_frame[:1], width, height, "center")
        images.append(img)
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

    tokens = clip.tokenize(prompt, images=images)
    cond = clip.encode_from_tokens_scheduled(tokens)

    for kf in keyframes:
        kf["latent"] = video_vae.encode(kf.pop("image"))
    if keyframes:
        cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
    return cond, latent, frame_count


def _add_lipsync_audio(cond, audio_vae, audio):
    """Anchors a dialogue track at frame 0 so the joint AV model drives the mouth.

    This is MiniMaxH3AddGuide's audio path: the audio latent is placed on the shared
    time axis and denoised together with the video, so speech in the guide lands on
    the same frames as the lip motion."""
    audio_latent, _rt = h3._encode_ref_audio(audio_vae, audio)
    keyframes = list(cond[0][1].get("minimax_keyframes", []))
    keyframes.append({"resolved_frame_index": 0, "audio_latent": audio_latent})
    return node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})


class GAPSegmentManager(io.ComfyNode):
    """Geekatplay Studio - Vladimir Chopine.
    Chains keyframe segments into one continuous FastH3 movie."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GAPSegmentManager",
            display_name="Segment Manager - Keyframe Chain (Geekatplay Studio)",
            category="Geekatplay/StoryTeller",
            description=(
                "Renders a chain of keyframes as one continuous video. Each connected "
                "segment's image is the opening frame of its own segment and the closing "
                "frame of the one before it. Connect segments in order; a new input slot "
                "appears as you fill the last one."),
            inputs=[
                io.Combo.Input("visual_style", options=VISUAL_STYLES, default="Cinematic Live-Action",
                               tooltip="The look, stamped identically onto every segment prompt."),
                io.Combo.Input("video_type", options=list(VIDEO_TYPES.keys()), default="Narrative Scene",
                               tooltip="What kind of film this is. Shapes framing and cutting language."),
                io.Combo.Input("pacing_and_mood", options=PACING_MOODS, default="Calm & Atmospheric",
                               tooltip="Lighting register. Only used by the two stronger "
                                       "style_mode settings."),
                io.Combo.Input("style_mode", options=STYLE_MODES,
                               default="none - send only my prompt (recommended)",
                               tooltip="How much text is appended to your prompt. The "
                                       "default appends NOTHING - what you typed is what is sent. "
                                       "The other settings add a look instruction of increasing "
                                       "strength, for when you want the render to depart from the "
                                       "source frames."),
                io.Combo.Input("aspect_ratio", options=ASPECT_CHOICES, default="match first keyframe",
                               tooltip="Screen SHAPE. 'match first keyframe' adopts the first "
                                       "image's proportions; anything else forces that ratio and "
                                       "the images are fitted to it."),
                io.Combo.Input("frame_size", options=list(FRAME_SIZES.keys()),
                               default="auto (MiniMax H3 native, ~1 MP)",
                               tooltip="Screen SIZE. How many pixels the shape is rendered at. "
                                       "Smaller renders much faster; 1344 is the most MiniMax H3 "
                                       "accepts on the long edge. Both axes snap to 32."),
                io.Int.Input("custom_width", default=1344, min=32, max=1344, step=32,
                             tooltip="Used only when frame_size is 'custom'."),
                io.Int.Input("custom_height", default=768, min=32, max=1344, step=32,
                             tooltip="Used only when frame_size is 'custom'."),
                io.Combo.Input("duration_source", options=DURATION_SOURCES,
                               default="from segment generator",
                               tooltip="Whether each segment uses the length its generator chose, or "
                                       "the manager's segment_duration."),
                io.Float.Input("segment_duration", default=5.0, min=3.0, max=10.0, step=0.5,
                               tooltip="Seconds per segment when the manager overrides or clamps. "
                                       "Snapped to the 17k+5 frame grid at 24 fps."),
                io.Boolean.Input("tail_segment", default=True,
                                 tooltip="Render one extra segment from the final keyframe with no "
                                         "closing frame, so the movie does not end on a hard freeze."),
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models"),
                               tooltip="FastVideo FastH3 8-step checkpoint."),
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders"),
                               tooltip="Qwen3-VL text encoder for MiniMax H3."),
                io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae"),
                               tooltip="MiniMax H3 video VAE."),
                io.Combo.Input("audio_vae_name", options=folder_paths.get_filename_list("vae"),
                               tooltip="MiniMax H3 audio VAE."),
                io.Int.Input("steps", default=8, min=1, max=100,
                             tooltip="8 for FastH3 8-step, 20 for base MiniMax H3."),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES, default="simple"),
                io.Float.Input("shift_video", default=10.0, min=0.01, max=100.0, step=0.01,
                               tooltip="MiniMax H3 video flow shift. Required for FastH3 to converge."),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01,
                               tooltip="MiniMax H3 audio flow shift."),
                io.Int.Input("noise_seed", default=42, min=0, max=0xffffffffffffffff,
                             control_after_generate=True,
                             tooltip="Base seed. Segment i uses noise_seed + i."),
                io.Autogrow.Input(
                    "segments",
                    tooltip="Connect Segment Generators in story order. Slot 1 is the first frame, "
                            "slot 2 is its closing frame and the next segment's opening frame, and so on.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Custom("GAP_SEGMENT").Input(
                            "segment", tooltip="Output of a Segment Generator."),
                        prefix="segment_", min=2, max=MAX_SEGMENTS)),
                io.Autogrow.Input(
                    "lipsync_audios", optional=True,
                    tooltip="EXPERIMENTAL. A dialogue track anchored at frame 0 of the same-numbered "
                            "segment, so MiniMax H3 animates the mouth against it.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("lipsync_audio",
                                             tooltip="Speech for the same-numbered segment."),
                        prefix="lipsync_audio_", min=0, max=MAX_SEGMENTS)),
                io.Audio.Input("background_music", optional=True,
                               tooltip="Optional music bed. Leave disconnected or bypassed to keep "
                                       "only the native MiniMax H3 audio."),
                io.Combo.Input("music_mix_mode", options=MUSIC_MIX_MODES,
                               default="blend with video audio", optional=True),
                io.Float.Input("music_gain", default=0.35, min=0.0, max=2.0, step=0.05, optional=True),
                io.Float.Input("sfx_gain", default=1.0, min=0.0, max=2.0, step=0.05, optional=True),
                io.Boolean.Input("trim_seam_frame", default=True, optional=True,
                                 tooltip="Drop the duplicate boundary frame between segments. Leave on: "
                                         "the shared keyframe is rendered at both ends of the seam."),
                io.Boolean.Input("save_segments", default=True, optional=True,
                                 tooltip="Checkpoint each finished segment to disk."),
                io.Boolean.Input("resume", default=True, optional=True,
                                 tooltip="Reuse checkpointed segments when nothing about them changed."),
                io.String.Input("segment_prefix", default="storyteller/KeyframeChain", optional=True,
                                tooltip="Where individual segments are written inside the ComfyUI "
                                        "output directory, as <prefix>_segment_001.mp4 and so on."),
                io.Boolean.Input("free_ollama_vram", default=True, optional=True,
                                 tooltip="Unload any model Ollama is holding before the video "
                                         "models load. The Segment Generators keep their LLM "
                                         "resident so they do not reload it per segment; this "
                                         "is where it gets freed."),
                io.String.Input("rerender_segments", default="", optional=True,
                                tooltip="Segment numbers to force a re-render of, e.g. '3' or "
                                        "'2, 5-7'. Everything else is reused from disk."),
                io.Boolean.Input("low_memory", default=True, optional=True,
                                 tooltip="Hold accumulated frames as 8-bit in RAM."),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Image.Output(display_name="frames"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="segment_paths"),
                io.String.Output(display_name="chain_report"),
                io.String.Output(display_name="segment_prompts"),
            ],
        )

    # ------------------------------------------------------------------ helpers
    @classmethod
    def _resolve_canvas(cls, aspect_ratio, first_image,
                        frame_size="auto (MiniMax H3 native, ~1 MP)",
                        custom_width=1344, custom_height=768):
        return resolve_canvas(aspect_ratio, frame_size, custom_width, custom_height,
                              image=first_image)

    @classmethod
    def _resolve_duration(cls, seg, duration_source, manager_seconds):
        seconds = float(seg.get("duration", 5.0))
        if duration_source == "manager override (all segments)":
            seconds = manager_seconds
        elif duration_source == "clamp generator to manager value":
            seconds = min(seconds, manager_seconds)
        frames = align_frame_count(seconds, FPS)
        return frames, round(frames / FPS, 2)

    @classmethod
    def _build_chain(cls, segments, tail_segment):
        """Pairs keyframe k with keyframe k+1. Returns render jobs in order."""
        jobs = []
        for k in range(len(segments) - 1):
            jobs.append({
                "source": segments[k],
                "first_frame": segments[k]["image"],
                "last_frame": segments[k + 1]["image"],
            })
        if tail_segment:
            jobs.append({
                "source": segments[-1],
                "first_frame": segments[-1]["image"],
                "last_frame": None,
            })
        return jobs

    # ------------------------------------------------------------------ execute
    @classmethod
    def execute(cls, visual_style, video_type, pacing_and_mood, style_mode, aspect_ratio,
                frame_size, custom_width, custom_height, duration_source,
                segment_duration, tail_segment, unet_name, clip_name, vae_name, audio_vae_name,
                steps, sampler_name, scheduler, shift_video, shift_audio, noise_seed,
                segments=None, lipsync_audios=None, background_music=None,
                music_mix_mode="blend with video audio", music_gain=0.35, sfx_gain=1.0,
                trim_seam_frame=True, save_segments=True, resume=True,
                segment_prefix="storyteller/KeyframeChain", free_ollama_vram=True,
                rerender_segments="", low_memory=True, **kwargs):

        if free_ollama_vram:
            _free_ollama_vram()

        seg_list = _ordered(segments)
        if len(seg_list) < 2:
            raise ValueError(
                "GAPSegmentManager needs at least two connected segments: the chain renders "
                "keyframe 1 to keyframe 2, keyframe 2 to keyframe 3, and so on.")
        for i, seg in enumerate(seg_list):
            if not isinstance(seg, dict) or seg.get("image") is None:
                raise ValueError(f"Segment input {i + 1} is not a GAP_SEGMENT from a Segment Generator.")

        lipsync = _ordered(lipsync_audios)

        width, height = cls._resolve_canvas(aspect_ratio, seg_list[0]["image"],
                                            frame_size, custom_width, custom_height)
        style_clause = _style_note(style_mode, visual_style, pacing_and_mood, video_type)
        jobs = cls._build_chain(seg_list, tail_segment)

        print(f"[Geekatplay Studio] Segment Manager: {len(seg_list)} keyframes -> {len(jobs)} segments "
              f"at {width}x{height}"
              f"{' (with tail segment)' if tail_segment else ''}.")

        model = _get_cached_model("model", unet_name, lambda: comfy.sd.load_diffusion_model(
            folder_paths.get_full_path_or_raise("diffusion_models", unet_name)))
        clip = _get_cached_model("clip", clip_name, lambda: comfy.sd.load_clip(
            ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", clip_name)],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.MINIMAX))
        video_vae = _get_cached_model("vae", vae_name, lambda: _load_vae(vae_name))
        audio_vae = _get_cached_model("audio_vae", audio_vae_name, lambda: _load_vae(audio_vae_name))
        model = _apply_sigma_shift(model, shift_video, shift_audio)

        all_frames, all_waves, segment_files, report, prompt_log = [], [], [], [], []
        sample_rate = None
        rerender_set = parse_rerender_list(rerender_segments, len(jobs))
        if rerender_set:
            print(f"[Geekatplay Studio] Forcing a re-render of segment(s): "
                  f"{', '.join(str(i) for i in sorted(rerender_set))}")
        pbar = comfy.utils.ProgressBar(len(jobs))

        for i, job in enumerate(jobs):
            seg = job["source"]
            has_last = job["last_frame"] is not None
            frame_count, duration = cls._resolve_duration(seg, duration_source, segment_duration)
            prompt = "{} {}".format(
                _rebuild_anchor(seg.get("shot_text", ""), has_last, duration,
                                wanted=bool(seg.get("frame_anchor_lines"))),
                style_clause).strip()
            seed = noise_seed + i
            voice = lipsync[i] if i < len(lipsync) else None

            fingerprint = _segment_fingerprint(
                unet=unet_name, clip=clip_name, vae=vae_name, audio_vae=audio_vae_name,
                prompt=prompt, width=width, height=height, frames=frame_count, steps=steps,
                sampler=sampler_name, scheduler=scheduler, seed=seed,
                shift_video=shift_video, shift_audio=shift_audio,
                first=_image_fingerprint(job["first_frame"]),
                last=_image_fingerprint(job["last_frame"]),
                lipsync=_audio_fingerprint(voice))
            path = _segment_path(segment_prefix, i + 1)
            forced = (i + 1) in rerender_set

            label = seg.get("label") or f"keyframe {i + 1}"

            # Exactly what this segment sends to MiniMax H3, so it can be read back
            # in the UI instead of guessed at.
            first_shape = tuple(int(x) for x in job["first_frame"].shape[1:3])
            last_shape = (tuple(int(x) for x in job["last_frame"].shape[1:3])
                          if has_last else None)
            prompt_log.append(
                f"{'=' * 72}\nSEGMENT {i + 1}/{len(jobs)}  -  {label}\n{'=' * 72}\n"
                f"first_frame : keyframe {i + 1}, source {first_shape[1]}x{first_shape[0]}"
                f"  -> anchored at frame 0\n"
                f"last_frame  : " + (
                    f"keyframe {i + 2}, source {last_shape[1]}x{last_shape[0]}"
                    f"  -> anchored at frame {frame_count - 1}"
                    if has_last else "none (tail segment, image-to-video)") + "\n"
                f"canvas      : {width}x{height}   frames: {frame_count} "
                f"({duration}s)   seed: {seed}\n"
                f"written by  : {seg.get('written_by', 'unknown')}\n"
                f"{'-' * 72}\n{prompt}\n")
            restored = _resume_segment(path, fingerprint, allowed=resume and not forced)
            if restored is not None:
                frames, wave, sr = restored
                print(f"[Geekatplay Studio] Reusing segment {i + 1}/{len(jobs)} ({label}).")
                report.append(f"[{i+1}] {label}: reused from disk, {duration}s")
            else:
                kind = ("first+last frame anchored" if has_last
                        else "first frame only (tail, i2v)")
                print(f"[Geekatplay Studio] Rendering segment {i + 1}/{len(jobs)} ({label}, {kind}, "
                      f"{frame_count} frames, seed {seed})...")
                cond, latent, _fc = _build_conditioning(
                    clip, video_vae, prompt, job["first_frame"], job["last_frame"],
                    width, height, frame_count)
                if voice is not None:
                    cond = _add_lipsync_audio(cond, audio_vae, voice)

                samples = _sample_segment(model, cond, latent, sampler_name, scheduler, steps, seed)
                frames, wave, sr = _decode_segment(samples, video_vae, audio_vae)

                del cond, latent, samples
                comfy.model_management.soft_empty_cache()

                if save_segments:
                    _save_segment(path, frames, wave, sr)
                    _write_sidecar(path, fingerprint, {
                        "segment": i + 1, "label": label, "duration": duration,
                        "frames": frame_count, "seed": seed, "prompt": prompt})
                report.append(f"[{i+1}] {label}: {kind}, {duration}s / {frame_count} frames, "
                              f"seed {seed}{', lipsync' if voice is not None else ''}")

            if save_segments:
                segment_files.append(path)
            sample_rate = sample_rate or sr

            # The shared keyframe is rendered as the last frame of this segment and the
            # first frame of the next, so one of the two copies has to go.
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
            base_wave=base_waveform, music_audio=background_music,
            num_frames=final_frames.shape[0], sample_rate=sample_rate,
            mode=music_mix_mode, music_gain=music_gain, sfx_gain=sfx_gain)

        audio_output = {"waveform": final_waveform, "sample_rate": sample_rate}
        final_video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=final_frames, audio=audio_output, frame_rate=Fraction(FPS)))

        total = final_frames.shape[0] / FPS
        header = (f"{len(seg_list)} keyframes -> {len(jobs)} segments -> {total:.1f}s "
                  f"at {width}x{height}\nStyle: {visual_style} | Type: {video_type} | "
                  f"Mood: {pacing_and_mood}")
        print(f"[Geekatplay Studio] Chain complete: {final_frames.shape[0]} frames ({total:.1f}s).")

        return io.NodeOutput(final_video, final_frames, audio_output,
                             "\n".join(segment_files), header + "\n\n" + "\n".join(report),
                             "\n".join(prompt_log))


def _image_fingerprint(image):
    """Cheap, stable summary of an image for the checkpoint fingerprint."""
    if image is None:
        return "none"
    t = image[:1].float()
    return f"{tuple(t.shape)}:{float(t.mean()):.6f}:{float(t.std()):.6f}"


def _audio_fingerprint(audio):
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return "none"
    w = audio["waveform"]
    return f"{tuple(w.shape)}:{int(audio.get('sample_rate', 0))}:{float(w.float().mean()):.6f}"
