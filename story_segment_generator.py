"""
Geekatplay Studio - Vladimir Chopine
Segment Generator for the StoryTeller Segment Chain
Website: https://www.geekatplay.com

Turns a keyframe's plain-language description into MiniMax H3 prompt syntax and
decides how long the segment should run, from 3s to 10s, based on how much has to
happen in it. The image passes straight through to the Segment Manager, which
chains it with the next keyframe as a first/last frame pair.
"""

import random
import re

from .story_director import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    FPS,
    SHOT_RECIPES,
    _call_ollama,
    _extract_json_block,
    _format_timecode,
    _load_reference_guides,
    _normalize_shot_text,
    align_frame_count,
    ollama_model_options,
    ollama_model_tooltip,
    resolve_ollama_model,
)

MIN_SEGMENT_SECONDS = 3.0
MAX_SEGMENT_SECONDS = 10.0

# Words that mean the shot has to cover ground. More of them, more seconds.
_COMPLEXITY_SIGNALS = {
    "motion": ["run", "runs", "running", "walk", "walks", "fly", "flies", "flying", "drive", "drives",
               "fall", "falls", "climb", "climbs", "jump", "jumps", "chase", "chases", "spin", "spins",
               "approach", "approaches", "cross", "crosses", "travel", "travels", "rise", "rises"],
    "transform": ["transform", "transforms", "morph", "morphs", "grow", "grows", "bloom", "blooms",
                  "dissolve", "shatter", "shatters", "unfold", "unfolds", "assemble", "assembles",
                  "awaken", "awakens", "ignite", "ignites", "collapse", "collapses", "emerge", "emerges"],
    "camera": ["pan", "tilt", "orbit", "crane", "push", "pull", "track", "zoom", "dolly", "arc", "sweep"],
    "beats": ["then", "before", "after", "while", "as", "until", "finally", "meanwhile", "suddenly"],
}


def estimate_duration(description, dialogue="", camera_hint="", floor=MIN_SEGMENT_SECONDS,
                      ceiling=MAX_SEGMENT_SECONDS):
    """Scores how much has to happen and maps it onto the 3s-10s range.

    A held portrait needs 3 seconds. A subject crossing a room while the camera
    orbits and a line is spoken needs closer to 8."""
    text = f"{description} {camera_hint}".lower()
    words = re.findall(r"[a-z']+", text)
    unique = set(words)

    score = 0.0
    score += min(3.0, len(words) / 22.0)                       # sheer amount described
    score += 1.4 * sum(1 for w in _COMPLEXITY_SIGNALS["motion"] if w in unique)
    score += 1.8 * sum(1 for w in _COMPLEXITY_SIGNALS["transform"] if w in unique)
    score += 0.8 * sum(1 for w in _COMPLEXITY_SIGNALS["camera"] if w in unique)
    score += 1.0 * sum(1 for w in _COMPLEXITY_SIGNALS["beats"] if w in unique)
    score += 0.9 * text.count(",")                              # clause count
    score += 1.6 * len(re.findall(r"[.;]", text.strip().rstrip(".")))   # final stop is not a beat

    if dialogue.strip():
        # Speech needs room to land: roughly 2.5 words a second plus a beat either side.
        spoken = len(re.findall(r"[\w']+", dialogue))
        score += max(2.0, spoken / 2.5)

    seconds = floor + (ceiling - floor) * min(1.0, score / 12.0)
    return round(min(ceiling, max(floor, seconds)) * 2) / 2.0   # snap to the half second


def snap_duration(seconds):
    """Snaps a duration onto MiniMax H3's 17k+5 frame grid and reports both."""
    frames = align_frame_count(seconds, FPS)
    return frames, round(frames / FPS, 2)


def _anchor_lines(has_last_frame, duration):
    """The frame-reference preamble, copied verbatim from the reference i2v workflow.

    Only the <Picture 1> sentence is real - it is what the official workflow ships. The
    matching "<Picture 2> at the end" sentence was an extrapolation of mine and is gone:
    the closing frame is anchored as a keyframe, not presented as a numbered picture."""
    return ("For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.")


# Camera moves phrased WITHOUT naming a subject. "orbits around the subject" tells the
# model there is a person to orbit; on a landscape it will invent one. The framing comes
# from the supplied frame, so no shot size is imposed either.
NEUTRAL_MOVES = [
    "pushes in slowly with small amplitude",
    "pulls back slowly with medium amplitude",
    "pans right with medium amplitude at slow speed",
    "pans left with medium amplitude at slow speed",
    "tilts up with small amplitude at slow speed",
    "tilts down with small amplitude at slow speed",
    "tracks forward with medium amplitude at medium speed",
    "cranes up with large amplitude at slow speed",
    "drifts laterally with small amplitude at slow speed",
    "holds static",
]


PROMPT_MODES = [
    "format my text only (no AI, no invention)",
    "format with Ollama (strict, invention rejected)",
    "expand creatively with Ollama",
]

# The documented MiniMax H3 motion vocabulary, and the words people actually type for it.
_CAMERA_VERBS = [
    (("pan",), "pans", ("left", "right")),
    (("tilt",), "tilts", ("up", "down")),
    (("push", "zoom in", "dolly in", "move in", "closer"), "pushes in", ()),
    (("pull", "zoom out", "dolly out", "move out", "back away"), "pulls back", ()),
    (("track", "follow", "dolly", "travel"), "tracks", ("forward", "backward", "left", "right")),
    (("orbit", "arc", "circle", "revolve", "rotate around"), "orbits", ("left", "right")),
    (("crane", "boom", "jib"), "cranes", ("up", "down")),
    (("handheld",), "follows handheld", ()),
    (("static", "locked", "hold", "still", "fixed"), "holds static", ()),
]
_AMPLITUDE = [(("slight", "subtle", "small", "gentle", "gently", "slightly", "tiny"), "small"),
              (("wide", "large", "big", "sweeping", "dramatic", "huge"), "large")]
_SPEED = [(("slow", "slowly", "gradual", "gradually", "smooth", "smoothly", "creep"), "slow"),
          (("fast", "quick", "quickly", "rapid", "rapidly", "snap", "whip", "sudden"), "fast")]


def format_camera_motion(text):
    """Turns plain words into 'motion type + amplitude + speed', or returns None.

    None means nothing camera-like was recognised, and the caller passes the user's
    sentence through untouched rather than guessing."""
    low = (text or "").lower()
    if not low.strip():
        return None

    verb = direction = None
    for triggers, canonical, directions in _CAMERA_VERBS:
        if any(t in low for t in triggers):
            verb = canonical
            for d in directions:
                if d in low or (d == "backward" and "back" in low):
                    direction = d
                    break
            break
    if verb is None:
        return None

    amplitude = "medium"
    for words, value in _AMPLITUDE:
        if any(w in low for w in words):
            amplitude = value
            break
    speed = "medium"
    for words, value in _SPEED:
        if any(w in low for w in words):
            speed = value
            break

    if verb == "holds static":
        return "The camera holds static."
    motion = f"{verb} {direction}" if direction else verb
    return f"The camera {motion} with {amplitude} amplitude at {speed} speed."


# Words that carry no subject of their own, so their presence does not make a clause
# "about" something other than the camera.
_FILLER = {
    "a", "an", "the", "it", "its", "this", "that", "there", "here", "then", "and", "or",
    "to", "of", "in", "on", "at", "by", "with", "from", "into", "onto", "over", "under",
    "around", "across", "through", "along", "toward", "towards", "past", "up", "down",
    "left", "right", "forward", "backward", "back", "out", "one", "another", "very",
    "shot", "camera", "cam", "view", "angle", "framing", "move", "moves", "moving",
    "motion", "speed", "amplitude", "is", "are", "be", "keep", "keeps", "stay", "stays",
}
_CAMERA_WORDS = set()
for _triggers, _canon, _dirs in _CAMERA_VERBS:
    _CAMERA_WORDS.update(w for t in _triggers for w in t.split())
for _words, _v in _AMPLITUDE + _SPEED:
    _CAMERA_WORDS.update(_words)


def is_camera_clause(fragment):
    """True only when the fragment is genuinely an instruction to the camera.

    Two ways to qualify: it names the camera, or removing every camera-vocabulary word
    and filler leaves nothing behind ("slow pan right", "hold still"). A clause like
    "a woman stands still" keeps "woman stands", so it is scene, not camera."""
    low = (fragment or "").lower()
    if not low.strip():
        return False
    if format_camera_motion(low) is None:
        return False
    words = re.findall(r"[a-z']+", low)
    if "camera" in words or "cam" in words:
        return True
    leftover = [w for w in words if w not in _CAMERA_WORDS and w not in _FILLER]
    return not leftover


def split_camera_clause(text):
    """Separates a sentence into (scene part, camera part).

    Splits on the connectors people actually use, then routes each fragment by whether
    it talks about the camera. Nothing is discarded: every fragment ends up in one side
    or the other."""
    raw = (text or "").strip()
    if not raw:
        return "", ""
    # Connectors people actually use to join a scene clause to a camera clause.
    fragments = re.split(
        r"\s*(?:,\s*and\s+|\s+and\s+|\s+while\s+|\s+whilst\s+|\s+as\s+the\s+camera\s+"
        r"|\s+with\s+the\s+camera\s+|,|;|\.)\s*", raw)
    scene, camera = [], []
    for frag in fragments:
        if not frag.strip():
            continue
        if is_camera_clause(frag):
            camera.append(frag.strip())
        else:
            scene.append(frag.strip())
    return ", ".join(scene), ", ".join(camera)


def _format_only_shot_text(description, camera_hint, dialogue, has_last_frame,
                           closing_frame_note=False):
    """Your words, in MiniMax H3 shape. Nothing added, nothing invented, nothing dropped.

    Returns the text plus a note saying exactly what was done to it."""
    parts = ["[Shot 1]"]
    note_bits = []

    if camera_hint.strip():
        # An explicit hint means the description is entirely scene.
        scene, camera_src = description.strip(), camera_hint.strip()
    else:
        scene, camera_src = split_camera_clause(description)

    if scene:
        parts.append(scene.rstrip(".") + ".")
        note_bits.append("your scene text kept verbatim")

    motion = format_camera_motion(camera_src) if camera_src else None
    if motion:
        parts.append(motion)
        note_bits.append(f"'{camera_src}' restated as: {motion}")
    elif camera_src:
        parts.append(camera_src.rstrip(".") + ".")
        note_bits.append("camera wording not recognised - kept verbatim")

    if not scene and not camera_src:
        note_bits.append("nothing to format")

    if has_last_frame and closing_frame_note:
        parts.append("The motion ends on the composition of the final frame.")
        note_bits.append("closing-frame line ADDED by you enabling closing_frame_note")

    if dialogue.strip():
        parts.append(f"(S1) says: <d>[English] {dialogue.strip().strip(chr(34))}</d>")
        note_bits.append("your dialogue wrapped in <d>[English]...</d>")

    return " ".join(parts), "; ".join(note_bits)


_CONTENT_WORD = re.compile(r"[a-z]{4,}")
_ALLOWED_EXTRA = {
    "shot", "camera", "amplitude", "speed", "slow", "medium", "fast", "small", "large",
    "pans", "tilts", "pushes", "pulls", "tracks", "orbits", "cranes", "holds", "static",
    "left", "right", "forward", "backward", "handheld", "follows", "with", "into", "that",
    "this", "then", "frame", "final", "composition", "ends", "motion", "english", "says",
    "toward", "across", "through", "while", "during", "onto", "from", "over", "under",
}


def invented_words(user_text, model_text):
    """Content words the model added that the user never wrote.

    This is the check that catches "dunes" appearing from a prompt that only said
    "pan slowly"."""
    allowed = set(_CONTENT_WORD.findall((user_text or "").lower())) | _ALLOWED_EXTRA
    return sorted({w for w in _CONTENT_WORD.findall((model_text or "").lower())
                   if w not in allowed})


def _procedural_shot_text(description, camera_hint, dialogue, duration, seed, has_last_frame):
    """Offline path: motion only, nothing that contradicts the supplied frames.

    Everything visible - who or what is in shot, the framing, the style - already comes
    from the first and last frame. This adds only how the camera and the scene move."""
    rng = random.Random(seed)
    move = camera_hint.strip() or NEUTRAL_MOVES[rng.randrange(len(NEUTRAL_MOVES))]
    body = description.strip().rstrip(".")

    text = "[Shot 1] " + (f"{body}. " if body else "")
    text += f"The camera {move}."
    if has_last_frame:
        text += (" The motion ends on the composition of the final frame, arriving on it "
                 "smoothly rather than cutting to it.")
    if dialogue.strip():
        line = dialogue.strip().strip('"')
        text += f" (S1) says: <d>[English] {line}</d>"
    return text


class GAPSegmentGenerator:
    """Geekatplay Studio - Vladimir Chopine.
    Rewrites a keyframe's plain description into MiniMax H3 prompt syntax and sets
    the segment length from scene complexity. Feeds the Segment Manager."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframe": ("GAP_KEYFRAME",),
                "prompt_mode": (PROMPT_MODES, {
                    "default": PROMPT_MODES[0],
                    "tooltip": "'format my text only' rewrites what you typed into MiniMax H3 "
                               "motion grammar and adds NOTHING - no AI is called, so it cannot "
                               "hallucinate. The strict Ollama mode formats with the LLM but "
                               "discards the result if it introduces words you never wrote. "
                               "'expand creatively' lets it write freely."
                }),
                "duration_mode": (["auto (from scene complexity)", "fixed"], {
                    "default": "auto (from scene complexity)",
                    "tooltip": "Auto reads the description and picks between min and max seconds. "
                               "Fixed always uses 'fixed_duration'."
                }),
                "min_seconds": ("FLOAT", {
                    "default": 3.0, "min": 3.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Shortest a generated segment may be."
                }),
                "max_seconds": ("FLOAT", {
                    "default": 10.0, "min": 3.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Longest a generated segment may be."
                }),
                "fixed_duration": ("FLOAT", {
                    "default": 5.0, "min": 3.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Used when duration_mode is 'fixed'."
                }),
            },
            "optional": {
                "is_final_keyframe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Turn on for the last keyframe in the chain. Its segment is rendered "
                               "as image-to-video with no closing frame. The manager also detects this "
                               "on its own, so it is normally left off."
                }),
                "closing_frame_note": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Append one sentence telling the model to land the motion on the "
                               "supplied final frame. OFF by default because it is text you did "
                               "not write. The last frame is already anchored as a keyframe "
                               "without it; turn this on only if your transitions overshoot."
                }),
                "frame_anchor_lines": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Prefix the prompt with a <Picture 1> frame-reference preamble. "
                               "OFF by default: <Picture N> is reference-to-video vocabulary, and "
                               "first/last frames are already anchored as keyframes without it. "
                               "Turn on only if you are copying a reference prompt that uses it."
                }),
                "use_ollama": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Rewrite the description with a local Ollama model. Falls back to the "
                               "procedural writer if the server is unreachable."
                }),
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (ollama_model_options(), {
                    "default": DEFAULT_OLLAMA_MODEL,
                    "tooltip": "Models on your Ollama are listed first, then ones worth "
                               "pulling. " + ollama_model_tooltip()
                }),
                "ollama_model_custom": ("STRING", {
                    "default": "",
                    "tooltip": "Any model name, used instead of the dropdown when filled in."
                }),
                "auto_pull_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Download the chosen model through Ollama if it is missing."
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Leave the LLM in VRAM between calls. With one generator per "
                               "keyframe this is the difference between loading the model once "
                               "and loading it for every segment. The Segment Manager frees it "
                               "before it starts rendering."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Reproducible staging and prompt wording."
                }),
            }
        }

    RETURN_TYPES = ("GAP_SEGMENT", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("segment", "prompt_preview", "duration", "frame_length")
    FUNCTION = "generate_segment"
    CATEGORY = "Geekatplay/StoryTeller"
    DESCRIPTION = ("Rewrites a keyframe description into MiniMax H3 prompt syntax and "
                   "picks a 3s-10s segment length from scene complexity.")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate_segment(self, keyframe, prompt_mode=PROMPT_MODES[0],
                         duration_mode="auto (from scene complexity)",
                         min_seconds=3.0, max_seconds=10.0, fixed_duration=5.0,
                         is_final_keyframe=False, closing_frame_note=False,
                         frame_anchor_lines=False, use_ollama=True,
                         ollama_url=DEFAULT_OLLAMA_URL, ollama_model=DEFAULT_OLLAMA_MODEL,
                         ollama_model_custom="", auto_pull_model=True,
                         keep_model_loaded=True, seed=42, **kwargs):

        if not isinstance(keyframe, dict) or keyframe.get("image") is None:
            raise ValueError("GAPSegmentGenerator needs a keyframe from GAPKeyframeInput.")

        description = keyframe.get("description", "")
        dialogue = keyframe.get("dialogue", "")
        camera_hint = keyframe.get("camera_hint", "")
        override = float(keyframe.get("duration_override", 0.0) or 0.0)

        floor, ceiling = min(min_seconds, max_seconds), max(min_seconds, max_seconds)

        if override > 0.0:
            seconds = min(ceiling, max(floor, override))
            reason = "keyframe override"
        elif duration_mode == "fixed":
            seconds = min(ceiling, max(floor, fixed_duration))
            reason = "fixed"
        else:
            seconds = estimate_duration(description, dialogue, camera_hint, floor, ceiling)
            reason = "scene complexity"

        frame_length, duration = snap_duration(seconds)

        # The generator writes the shot; the Segment Manager owns the style and stamps
        # it on later, so the same shot text can be reused under any look.
        shot_text = None
        written_by = "procedural writer (Ollama switched off)"
        format_note = ""

        if prompt_mode == PROMPT_MODES[0]:
            shot_text, format_note = _format_only_shot_text(
                description, camera_hint, dialogue, not is_final_keyframe, closing_frame_note)
            written_by = "your text, formatted (no AI)"
            use_ollama = False

        if use_ollama:
            resolved_model = resolve_ollama_model(
                ollama_url, ollama_model, ollama_model_custom, auto_pull_model)
            if resolved_model is None:
                written_by = "procedural writer (no Ollama reachable)"
            else:
                shot_text = self._ollama_shot(description, dialogue, camera_hint, duration,
                                              not is_final_keyframe, ollama_url,
                                              resolved_model, seed, keep_model_loaded)
                if shot_text and prompt_mode == PROMPT_MODES[1]:
                    added = invented_words(f"{description} {camera_hint} {dialogue}", shot_text)
                    if added:
                        print(f"[Geekatplay Studio] {resolved_model} introduced words you never "
                              f"wrote {added[:8]} - discarding it and formatting your text "
                              f"instead. Use 'expand creatively' if you want it to invent.")
                        shot_text, format_note = _format_only_shot_text(
                            description, camera_hint, dialogue, not is_final_keyframe,
                            closing_frame_note)
                        written_by = f"your text, formatted ({resolved_model} rejected: {added[:4]})"
                    else:
                        written_by = f"Ollama ({resolved_model}, verified no invention)"
                else:
                    written_by = (f"Ollama ({resolved_model})" if shot_text
                                  else f"procedural writer ({resolved_model} failed)")
        if not shot_text:
            shot_text = _procedural_shot_text(description, camera_hint, dialogue, duration,
                                              seed, not is_final_keyframe)

        shot_text = _normalize_shot_text(shot_text, round(duration * 0.55, 1))
        if frame_anchor_lines:
            shot_text = f"{_anchor_lines(not is_final_keyframe, duration)} {shot_text}"

        segment = {
            "image": keyframe["image"],
            "shot_text": shot_text,
            "description": description,
            "dialogue": dialogue,
            "label": keyframe.get("label", ""),
            "duration": duration,
            "frame_length": frame_length,
            "duration_reason": reason,
            "written_by": written_by,
            "is_final_keyframe": bool(is_final_keyframe),
            "frame_anchor_lines": bool(frame_anchor_lines),
            "seed": int(seed),
        }

        preview = (f"[length {duration}s / {frame_length} frames, from {reason}]\n"
                   f"[prompt written by {written_by}]\n"
                   + (f"[what was done: {format_note}]\n" if format_note else "")
                   + f"\n{shot_text}")
        return (segment, preview, duration, frame_length)

    # ------------------------------------------------------------------ Ollama
    def _ollama_shot(self, description, dialogue, camera_hint, duration, has_last_frame,
                     url, model, seed, keep_loaded=True):
        base_guide, ref_guide = _load_reference_guides()
        closing = ("This segment ends on a supplied final frame, so the action must ARRIVE at that "
                   "composition naturally by the end of the shot - never cut to it."
                   if has_last_frame else
                   "This segment has no supplied final frame. End it on a stable, settled pose.")
        cut_rule = (f"Include a timed mid-shot cut written as '[Shot 2] At "
                    f"{_format_timecode(round(duration * 0.55, 1))}, the camera cuts to ...'."
                    if duration >= 6.0 else
                    "Keep this to a single continuous shot. Do not write a [Shot 2].")
        speech = (f"The subject speaks this line, exactly once, formatted as "
                  f"'The subject (S1) says: <d>[English] {dialogue.strip()}</d>'."
                  if dialogue.strip() else "No dialogue in this shot.")
        camera = (f"Use this camera instruction: {camera_hint}." if camera_hint.strip()
                  else "Choose one camera move and write it with motion type + amplitude + "
                       "speed, without naming what it moves around.")

        instruction = f"""You write single-shot prompts for MiniMax H3 image-to-video.

=== BASE PROMPTING GUIDE (base-en.txt) ===
{base_guide}

=== REFERENCE GUIDE (ref-en.txt) ===
{ref_guide}

This is IMAGE TO VIDEO. A real first frame is supplied and the model can already see it.
Everything visible is decided by that frame: what is in shot, who is in shot (possibly
nobody), the framing, the lighting and the style.

Your ONLY job is to describe how the scene and the camera MOVE between the frames.

Hard rules:
- Never invent a person, an object or a location that the description does not mention.
  If the description is a landscape, there is no person in it.
- Never name a shot size ("medium shot", "over-the-shoulder", "close-up"). The framing is
  already set by the supplied frame; naming a different one fights it. An over-the-shoulder
  shot in particular tells the model to put a person in frame.
- Never write "the subject" unless the description names one.
- Do not describe style, colour grade, lens or wardrobe. The frame already has them.
- Open with "[Shot 1] " and write one continuous present-tense paragraph.
- {camera}
- {cut_rule}
- {closing}
- {speech}
- The shot runs {duration} seconds. Do not describe more motion than fits.
- No second person, no instructions, no "the video shows". No on-screen text.

Respond ONLY with JSON: {{"shot": "..."}}"""

        user = f"Shot description from the user:\n{description}"

        try:
            raw = _call_ollama(url, model, f"{instruction}\n\nUser Request:\n{user}\n\nJSON Response:",
                               seed=seed, temperature=0.9,
                               keep_alive="15m" if keep_loaded else 0)
            parsed = _extract_json_block(raw)
            shot = (parsed.get("shot") or "").strip()
            if shot:
                print(f"[Geekatplay Studio] Segment Generator: Ollama wrote a {duration}s shot.")
                return shot
            print("[Geekatplay Studio] Segment Generator: Ollama returned no shot, using the "
                  "procedural writer.")
        except Exception as exc:
            print(f"[Geekatplay Studio] Segment Generator: Ollama unavailable ({exc}), using the "
                  "procedural writer.")
        return None
