"""
Geekatplay Studio - Vladimir Chopine
Lipsync Splitter & Renderer for MiniMax H3
Website: https://www.geekatplay.com

Takes a video and a dialogue track of any length, cuts the timeline into segments
that MiniMax H3 can actually render, re-generates each one with the speech anchored
as an audio guide, and puts the result back together.

Two things make the split non-trivial:

  1. MiniMax H3 only accepts frame counts on the 17k + 5 grid, so segment lengths are
     not free - they come from a discrete set (5, 22, 39 ... 124, 141, 158 ...).
  2. A cut in the middle of a spoken word is audible and looks wrong.

So the splitter does not cut at a fixed interval. For each segment it takes every
grid-legal length inside the allowed range, scores where each one would land - inside
a word is bad, inside a pause is good - and takes the best. Word timings come from
faster-whisper when it is installed, and from the audio's own energy envelope when it
is not.
"""

from fractions import Fraction

import numpy as np
import torch
import torchaudio

import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
from comfy_api.latest import InputImpl, Types, io
from comfy_extras import nodes_minimax_h3 as h3

from .story_director import free_ollama_vram as _free_ollama_vram
from .gap_render_core import (
    FPS,
    _apply_sigma_shift,
    _decode_segment,
    _get_cached_model,
    _load_vae,
    _resume_segment,
    _sample_segment,
    _save_segment,
    _segment_fingerprint,
    _segment_path,
    _write_sidecar,
    parse_rerender_list,
)

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
SPLIT_METHODS = [
    "auto (whisper words, else silence)",
    "whisper word timestamps",
    "silence detection",
    "fixed length (may cut words)",
]
ASR_SAMPLE_RATE = 16000
_ASR_CACHE = {}


# --------------------------------------------------------------------- grid
def grid_lengths(min_seconds, max_seconds, fps=FPS):
    """Every frame count MiniMax H3 accepts inside the allowed second range."""
    lengths = []
    k = 0
    while True:
        n = 17 * k + 5
        secs = n / fps
        if secs > max_seconds:
            break
        if secs >= min_seconds:
            lengths.append(n)
        k += 1
    if not lengths:                       # range narrower than the gap between two grid steps
        n = 5
        while n / fps < min_seconds:
            n += 17
        lengths.append(n)
    return lengths


# ----------------------------------------------------------------- word timing
def _mono_16k(waveform, sample_rate):
    wave = waveform
    if wave.ndim == 3:
        wave = wave[0]
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sample_rate != ASR_SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sample_rate, ASR_SAMPLE_RATE)
    return wave[0].detach().cpu().float().numpy()


def transcribe_words(waveform, sample_rate, model_size="base", language="auto"):
    """Word-level timings from faster-whisper. Returns [] when it is not installed."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[Geekatplay Studio] faster-whisper is not installed; falling back to silence "
              "detection for the split points. (pip install faster-whisper)")
        return []

    try:
        key = (model_size, "cuda" if torch.cuda.is_available() else "cpu")
        if key not in _ASR_CACHE:
            device = key[1]
            _ASR_CACHE[key] = WhisperModel(
                model_size, device=device,
                compute_type="float16" if device == "cuda" else "int8")
        model = _ASR_CACHE[key]

        audio = _mono_16k(waveform, sample_rate)
        segments, _info = model.transcribe(
            audio, word_timestamps=True,
            language=None if language.strip().lower() in ("", "auto") else language.strip())

        words = []
        for seg in segments:
            for w in (seg.words or []):
                words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
        print(f"[Geekatplay Studio] Transcribed {len(words)} words with faster-whisper "
              f"({model_size}).")
        return words
    except Exception as exc:
        print(f"[Geekatplay Studio] Transcription failed ({exc}); falling back to silence detection.")
        return []


def energy_envelope(waveform, sample_rate, hop_seconds=0.01):
    """Short-time RMS of the track, one value every hop_seconds."""
    wave = waveform
    if wave.ndim == 3:
        wave = wave[0]
    mono = wave.mean(dim=0).detach().cpu().float()
    hop = max(1, int(round(sample_rate * hop_seconds)))
    win = hop * 3
    if mono.shape[0] < win:
        return np.zeros(1, dtype=np.float32), hop_seconds
    frames = mono.unfold(0, win, hop)
    rms = frames.pow(2).mean(dim=1).sqrt().numpy()
    return rms, hop_seconds


# ------------------------------------------------------------- cut scoring
def _word_penalty(words, t):
    """0 when t sits in a pause, 1 when it lands in the middle of a word."""
    for w in words:
        if w["start"] <= t <= w["end"]:
            span = max(1e-3, w["end"] - w["start"])
            middle = 1.0 - abs((t - w["start"]) / span - 0.5) * 2.0   # 1 at the centre
            return 0.45 + 0.55 * middle
    return 0.0


def _gap_bonus(words, t):
    """How much clear air surrounds t, in seconds, capped."""
    before = [w["end"] for w in words if w["end"] <= t]
    after = [w["start"] for w in words if w["start"] >= t]
    if not before or not after:
        return 0.5
    return min(0.5, (min(after) - max(before)) / 2.0)


def _energy_penalty(rms, hop, t):
    if rms.size == 0:
        return 0.0
    i = min(rms.size - 1, max(0, int(round(t / hop))))
    peak = float(rms.max()) or 1.0
    return float(rms[i]) / peak


def choose_cuts(total_frames, words, rms, hop, min_seconds, max_seconds, target_seconds,
                fps=FPS, fixed=False):
    """Walks the timeline picking grid-legal segment lengths that land in the quiet.

    Returns a list of (start_frame, frame_count, quality) where quality is 1.0 for a
    clean cut in a pause and drops toward 0 for one inside a word."""
    lengths = grid_lengths(min_seconds, max_seconds, fps)
    target_frames = target_seconds * fps
    cuts = []
    start = 0

    while start < total_frames:
        remaining = total_frames - start
        if remaining <= lengths[-1]:
            # Last segment: smallest grid length that covers what is left. The render is
            # trimmed back to `remaining` afterwards, so no padding reaches the output.
            pick = next((n for n in lengths if n >= remaining), lengths[-1])
            cuts.append((start, pick, 1.0, remaining))
            break

        best, best_score = None, None
        for n in lengths:
            if fixed:
                score = -abs(n - target_frames)
            else:
                t = (start + n) / fps
                penalty = _word_penalty(words, t) if words else _energy_penalty(rms, hop, t)
                drift = abs(n - target_frames) / max(1.0, target_frames)
                score = -(penalty * 3.0) - drift + (_gap_bonus(words, t) if words else 0.0)
            if best_score is None or score > best_score:
                best, best_score = n, score

        t = (start + best) / fps
        penalty = _word_penalty(words, t) if words else _energy_penalty(rms, hop, t)
        cuts.append((start, best, round(1.0 - min(1.0, penalty), 2), best))
        start += best

    return cuts


def _slice_audio(waveform, sample_rate, start_frame, frame_count, fps=FPS):
    """The audio under a frame range, zero-padded if the track ends early."""
    a = int(round(start_frame * sample_rate / fps))
    n = int(round(frame_count * sample_rate / fps))
    wave = waveform
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    chunk = wave[..., a:a + n]
    if chunk.shape[-1] < n:
        chunk = torch.nn.functional.pad(chunk, (0, n - chunk.shape[-1]))
    return chunk


def _words_in(words, t0, t1):
    return " ".join(w["word"] for w in words if w["start"] >= t0 - 0.01 and w["end"] <= t1 + 0.01)


# =====================================================================
class GAPLipsyncSplitter:
    """Geekatplay Studio - Vladimir Chopine.
    Cuts a video and its dialogue into MiniMax H3 sized segments without cutting
    through a spoken word."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The source video to lipsync."}),
                "split_method": (SPLIT_METHODS, {
                    "default": "auto (whisper words, else silence)",
                    "tooltip": "How the cut points are found. Auto uses faster-whisper word "
                               "timings when it is installed, otherwise the audio's energy."
                }),
                "target_seconds": ("FLOAT", {
                    "default": 6.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Preferred segment length. The splitter moves off this to avoid "
                               "cutting a word."
                }),
                "min_seconds": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 15.0, "step": 0.5,
                    "tooltip": "Shortest allowed segment."
                }),
                "max_seconds": ("FLOAT", {
                    "default": 9.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Longest allowed segment. Above ~10s MiniMax H3 quality drops."
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Dialogue track. Leave disconnected to use the video's own audio."
                }),
                "whisper_model": (WHISPER_SIZES, {
                    "default": "base",
                    "tooltip": "faster-whisper size. 'base' is enough for word boundaries; "
                               "larger is slower and only helps the transcript text."
                }),
                "language": ("STRING", {
                    "default": "auto",
                    "tooltip": "Spoken language code, e.g. 'en'. 'auto' detects it."
                }),
            }
        }

    RETURN_TYPES = ("GAP_LIPSYNC_PLAN", "STRING", "INT", "IMAGE")
    RETURN_NAMES = ("plan", "split_report", "segment_count", "segment_thumbnails")
    FUNCTION = "split"
    CATEGORY = "Geekatplay/StoryTeller"
    DESCRIPTION = "Splits a video and its speech into MiniMax H3 segments on word boundaries."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def split(self, video, split_method="auto (whisper words, else silence)", target_seconds=6.0,
              min_seconds=4.0, max_seconds=9.0, audio=None, whisper_model="base",
              language="auto", **kwargs):

        components = video.get_components()
        frames = components.images
        src_fps = float(components.frame_rate) if components.frame_rate else FPS
        total_frames = int(frames.shape[0])

        track = audio if isinstance(audio, dict) and audio.get("waveform") is not None \
            else components.audio
        if not isinstance(track, dict) or track.get("waveform") is None:
            raise ValueError("Lipsync needs an audio track: connect one, or use a video that has one.")

        waveform = track["waveform"].detach().clone().float()
        sample_rate = int(track["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        # MiniMax H3 renders at 24 fps. Resample the timeline so a frame index means the
        # same instant in the source and in the render.
        if abs(src_fps - FPS) > 0.01:
            idx = torch.linspace(0, total_frames - 1, int(round(total_frames * FPS / src_fps)))
            frames = frames[idx.round().long().clamp(0, total_frames - 1)]
            total_frames = int(frames.shape[0])
            print(f"[Geekatplay Studio] Source is {src_fps:.2f} fps; resampled the frame timeline "
                  f"to {FPS} fps ({total_frames} frames).")

        duration = total_frames / FPS
        if min_seconds > max_seconds:
            min_seconds, max_seconds = max_seconds, min_seconds

        words = []
        used = "silence detection"
        if split_method in ("auto (whisper words, else silence)", "whisper word timestamps"):
            words = transcribe_words(waveform, sample_rate, whisper_model, language)
            if words:
                used = f"whisper word timestamps ({whisper_model})"
            elif split_method == "whisper word timestamps":
                print("[Geekatplay Studio] Whisper produced no words; using silence detection.")

        rms, hop = energy_envelope(waveform, sample_rate)
        fixed = split_method == "fixed length (may cut words)"
        if fixed:
            used = "fixed length"

        cuts = choose_cuts(total_frames, words, rms, hop, min_seconds, max_seconds,
                           target_seconds, FPS, fixed=fixed)

        segments = []
        thumbs = []
        for i, (start, frame_count, quality, keep) in enumerate(cuts):
            end = min(total_frames, start + frame_count)
            first = frames[start:start + 1]
            last_idx = min(total_frames - 1, start + frame_count - 1)
            seg_frames = frames[start:end]
            if seg_frames.shape[0] < frame_count:        # pad the final short segment
                pad = frame_count - seg_frames.shape[0]
                seg_frames = torch.cat([seg_frames, seg_frames[-1:].repeat(pad, 1, 1, 1)], dim=0)

            t0, t1 = start / FPS, (start + keep) / FPS
            segments.append({
                "index": i + 1,
                "start_frame": start,
                "frame_count": frame_count,
                "keep_frames": keep,
                "duration": round(frame_count / FPS, 2),
                "kept_duration": round(keep / FPS, 2),
                "start_time": round(t0, 2),
                "end_time": round(t1, 2),
                "first_frame": first,
                "last_frame": frames[last_idx:last_idx + 1],
                "source_frames": seg_frames,
                "audio": {"waveform": _slice_audio(waveform, sample_rate, start, frame_count),
                          "sample_rate": sample_rate},
                "text": _words_in(words, t0, t1) if words else "",
                "cut_quality": quality,
            })
            thumbs.append(first)

        plan = {
            "segments": segments,
            "total_frames": total_frames,
            "fps": FPS,
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "full_audio": {"waveform": waveform, "sample_rate": sample_rate},
            "method": used,
            "words": words,
        }

        lines = [
            f"Source: {duration:.2f}s, {total_frames} frames at {FPS} fps",
            f"Split method: {used}",
            f"Segments: {len(segments)} (target {target_seconds}s, allowed "
            f"{min_seconds}-{max_seconds}s)",
            "",
            f"{'#':>3}  {'start':>7}  {'end':>7}  {'len':>6}  {'frames':>6}  cut   text",
        ]
        for s in segments:
            flag = "clean" if s["cut_quality"] >= 0.85 else (
                "ok" if s["cut_quality"] >= 0.6 else "IN WORD")
            lines.append(f"{s['index']:>3}  {s['start_time']:>7.2f}  {s['end_time']:>7.2f}  "
                         f"{s['kept_duration']:>6.2f}  {s['frame_count']:>6}  {flag:<7} "
                         f"{s['text'][:60]}")
        rough = [s["index"] for s in segments if s["cut_quality"] < 0.6]
        if rough:
            lines += ["", f"Cuts that could not avoid a word: {rough}. Widen min/max seconds "
                          f"or nudge target_seconds to give the splitter more room."]

        report = "\n".join(lines)
        print(f"[Geekatplay Studio] Lipsync split: {len(segments)} segments over {duration:.1f}s "
              f"via {used}.")
        return (plan, report, len(segments), torch.cat(thumbs, dim=0))


# =====================================================================
class GAPLipsyncRenderer:
    """Geekatplay Studio - Vladimir Chopine.
    Re-renders every segment of a lipsync plan with the speech anchored as an audio
    guide, then reassembles the full-length video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("GAP_LIPSYNC_PLAN",),
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                "vae_name": (folder_paths.get_filename_list("vae"), ),
                "audio_vae_name": (folder_paths.get_filename_list("vae"), ),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The speaker talks directly to camera, lips and jaw matching the "
                               "spoken words, natural micro-expressions, head held steady.",
                    "tooltip": "Describes the performance, not the identity - the identity comes "
                               "from the source frames."
                }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "shift_video": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "noise_seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True}),
            },
            "optional": {
                "anchor_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also anchor the source segment's final frame, so each segment "
                               "lands where the next one starts and the joins do not jump."
                }),
                "keep_source_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use the original dialogue track for the finished video instead of "
                               "the model's generated audio. Leave on: it guarantees the words "
                               "stay exactly where they were."
                }),
                "segment_prefix": ("STRING", {
                    "default": "lipsync/MiniMaxH3",
                    "tooltip": "Individual segments are written here as "
                               "<prefix>_segment_001.mp4 inside the output directory."
                }),
                "rerender_segments": ("STRING", {
                    "default": "",
                    "tooltip": "Segment numbers to force a re-render of, e.g. '3' or '2, 5-7'. "
                               "Everything else is reused from disk. This is how you fix one bad "
                               "segment without redoing the whole video."
                }),
                "free_ollama_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload any model Ollama is holding before the video models load. "
                               "Whisper and MiniMax H3 both want the card."
                }),
                "save_segments": ("BOOLEAN", {"default": True}),
                "resume": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frames", "audio", "segment_paths", "render_report",
                    "segment_thumbnails", "segment_prompts")
    FUNCTION = "render"
    CATEGORY = "Geekatplay/StoryTeller"
    DESCRIPTION = "Renders a lipsync plan segment by segment and reassembles it."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def render(self, plan, unet_name, clip_name, vae_name, audio_vae_name, prompt,
               steps=8, sampler_name="euler", scheduler="simple", shift_video=10.0,
               shift_audio=3.0, noise_seed=42, anchor_last_frame=True, keep_source_audio=True,
               segment_prefix="lipsync/MiniMaxH3", rerender_segments="",
               free_ollama_vram=True, save_segments=True, resume=True, low_memory=True,
               **kwargs):

        if not isinstance(plan, dict) or not plan.get("segments"):
            raise ValueError("GAPLipsyncRenderer needs a plan from GAPLipsyncSplitter.")

        if free_ollama_vram:
            _free_ollama_vram()

        segments = plan["segments"]
        sample_rate = plan["sample_rate"]
        first_img = segments[0]["first_frame"]
        width, height = h3.adapt_canvas(int(first_img.shape[2]), int(first_img.shape[1]))

        rerender_set = parse_rerender_list(rerender_segments, len(segments))
        if rerender_set:
            print(f"[Geekatplay Studio] Forcing a re-render of segment(s): "
                  f"{', '.join(str(i) for i in sorted(rerender_set))}")

        model = _get_cached_model("model", unet_name, lambda: comfy.sd.load_diffusion_model(
            folder_paths.get_full_path_or_raise("diffusion_models", unet_name)))
        clip = _get_cached_model("clip", clip_name, lambda: comfy.sd.load_clip(
            ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", clip_name)],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.MINIMAX))
        video_vae = _get_cached_model("vae", vae_name, lambda: _load_vae(vae_name))
        audio_vae = _get_cached_model("audio_vae", audio_vae_name, lambda: _load_vae(audio_vae_name))
        model = _apply_sigma_shift(model, shift_video, shift_audio)

        print(f"[Geekatplay Studio] Lipsync render: {len(segments)} segments at {width}x{height}.")

        out_frames, out_waves, files, report, thumbs, prompt_log = [], [], [], [], [], []
        pbar = comfy.utils.ProgressBar(len(segments))

        for seg in segments:
            idx = seg["index"]
            seg_prompt = prompt.strip()
            if seg.get("text"):
                seg_prompt = (f"{seg_prompt} The speaker says: "
                              f"<d>[English] {seg['text']}</d>")

            prompt_log.append(
                f"{'=' * 72}\nSEGMENT {idx}/{len(segments)}  "
                f"{seg['start_time']:.2f}s - {seg['end_time']:.2f}s\n{'=' * 72}\n"
                f"first_frame : source frame {seg['start_frame']} -> anchored at frame 0\n"
                f"last_frame  : " + ("source frame "
                                     f"{seg['start_frame'] + seg['frame_count'] - 1} -> anchored "
                                     f"at frame {seg['frame_count'] - 1}"
                                     if anchor_last_frame else "not anchored") + "\n"
                f"audio guide : {seg['audio']['waveform'].shape[-1]} samples at "
                f"{sample_rate} Hz -> anchored at frame 0\n"
                f"canvas      : {width}x{height}   frames: {seg['frame_count']}   "
                f"seed: {noise_seed + idx}\n{'-' * 72}\n{seg_prompt}\n")

            fingerprint = _segment_fingerprint(
                unet=unet_name, clip=clip_name, vae=vae_name, audio_vae=audio_vae_name,
                prompt=seg_prompt, width=width, height=height, frames=seg["frame_count"],
                steps=steps, sampler=sampler_name, scheduler=scheduler,
                seed=noise_seed + idx, shift_video=shift_video, shift_audio=shift_audio,
                anchor_last=anchor_last_frame, start=seg["start_frame"])
            path = _segment_path(segment_prefix, idx)
            forced = idx in rerender_set

            restored = _resume_segment(path, fingerprint, allowed=resume and not forced)
            if restored is not None:
                frames, wave, sr = restored
                print(f"[Geekatplay Studio] Reusing lipsync segment {idx}/{len(segments)} "
                      f"({seg['start_time']:.2f}s - {seg['end_time']:.2f}s).")
                status = "reused"
            else:
                print(f"[Geekatplay Studio] Rendering lipsync segment {idx}/{len(segments)} "
                      f"({seg['start_time']:.2f}s - {seg['end_time']:.2f}s, "
                      f"{seg['frame_count']} frames)...")
                cond, latent = self._conditioning(
                    clip, video_vae, audio_vae, seg_prompt, seg, width, height,
                    anchor_last_frame)
                samples = _sample_segment(model, cond, latent, sampler_name, scheduler,
                                          steps, noise_seed + idx)
                frames, wave, sr = _decode_segment(samples, video_vae, audio_vae)
                del cond, latent, samples
                comfy.model_management.soft_empty_cache()

                if save_segments:
                    _save_segment(path, frames, wave, sr)
                    _write_sidecar(path, fingerprint, {
                        "segment": idx, "start_time": seg["start_time"],
                        "end_time": seg["end_time"], "frames": seg["frame_count"],
                        "cut_quality": seg["cut_quality"], "text": seg.get("text", ""),
                        "seed": noise_seed + idx})
                status = "rendered"

            if save_segments:
                files.append(path)

            # Segments are disjoint slices of one timeline, so nothing is trimmed at the
            # joins - trimming here would shift every later word off its frame.
            keep = min(seg["keep_frames"], frames.shape[0])
            frames = frames[:keep]
            thumbs.append(frames[:1])

            if keep_source_audio:
                wave = _slice_audio(plan["full_audio"]["waveform"], sample_rate,
                                    seg["start_frame"], keep)
            else:
                if sr != sample_rate:
                    wave = torchaudio.functional.resample(wave, sr, sample_rate)
                wave = wave[..., :int(round(keep * sample_rate / FPS))]

            if low_memory:
                frames = (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

            out_frames.append(frames)
            out_waves.append(wave)
            report.append(
                f"[{idx}] {seg['start_time']:>7.2f}s - {seg['end_time']:>7.2f}s  "
                f"{keep:>4}f  cut {seg['cut_quality']:.2f}  {status:<8} "
                f"{seg.get('text', '')[:48]}")
            pbar.update(1)

        final_frames = torch.cat(out_frames, dim=0)
        out_frames.clear()
        if final_frames.dtype == torch.uint8:
            final_frames = final_frames.float().div_(255.0)

        min_ch = min(w.shape[1] for w in out_waves)
        waveform = torch.cat([w[:, :min_ch] for w in out_waves], dim=-1)
        out_waves.clear()

        audio_out = {"waveform": waveform, "sample_rate": sample_rate}
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=final_frames, audio=audio_out, frame_rate=Fraction(FPS)))

        header = (f"Lipsync: {len(segments)} segments, {final_frames.shape[0]} frames "
                  f"({final_frames.shape[0] / FPS:.2f}s) at {width}x{height}\n"
                  f"Source was {plan['duration']:.2f}s / {plan['total_frames']} frames | "
                  f"audio: {'source track' if keep_source_audio else 'model generated'}")
        print(f"[Geekatplay Studio] Lipsync complete: {final_frames.shape[0]} frames "
              f"({final_frames.shape[0] / FPS:.2f}s).")

        return (video, final_frames, audio_out, "\n".join(files),
                header + "\n\n" + "\n".join(report), torch.cat(thumbs, dim=0),
                "\n".join(prompt_log))

    # ------------------------------------------------------------------
    def _conditioning(self, clip, video_vae, audio_vae, prompt, seg, width, height,
                      anchor_last):
        latent, frame_count = h3._empty_av_latent(width, height, seg["frame_count"])

        images, keyframes = [], []
        first = h3._resize(seg["first_frame"], width, height, "center")
        images.append(first)
        keyframes.append({"resolved_frame_index": 0, "image": first})
        if anchor_last:
            last = h3._resize(seg["last_frame"], width, height, "center")
            images.append(last)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": last})

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        for kf in keyframes:
            kf["latent"] = video_vae.encode(kf.pop("image"))

        # The dialogue itself, on the shared audio/video time axis at frame 0. This is
        # what makes the mouth follow the words rather than inventing speech.
        audio_latent, _rt = h3._encode_ref_audio(audio_vae, seg["audio"])
        keyframes.append({"resolved_frame_index": 0, "audio_latent": audio_latent})

        cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
        return cond, latent
