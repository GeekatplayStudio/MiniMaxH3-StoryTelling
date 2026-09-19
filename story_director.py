"""
Geekatplay Studio - Vladimir Chopine
StoryDirector for MiniMax FastH3 Lite & ComfyUI
Website: https://www.geekatplay.com

Directs multi-segment cinematic video generation by leveraging local Ollama
and the official MiniMax H3 reference guidelines (base-en.txt & ref-en.txt).
Maintains character consistency, camera motion vocabulary, soundscapes, and dialogue.
"""

import json
import math
import os
import random
import re
import time
import urllib.request
import urllib.error

FPS = 24

ASPECT_RATIOS = {
    "16:9 (Landscape)": (16, 9),
    "9:16 (Portrait)": (9, 16),
    "1:1 (Square)": (1, 1),
    "4:3 (Classic)": (4, 3),
    "3:4 (Classic Portrait)": (3, 4),
    "21:9 (Cinematic)": (21, 9),
}

VISUAL_STYLES = [
    "Cinematic Live-Action", "Anime / Studio Ghibli", "3D Pixar / Disney CGI",
    "Dark Fantasy / Gritty Gothic", "Cyberpunk Neon Noir", "80s Retro Sci-Fi Synthwave",
    "Watercolor Storybook", "Vintage 70s Panavision Film", "Claymation / Stop-Motion",
    "Graphic Novel / High-Contrast Comic", "Custom",
]

PACING_MOODS = [
    "Whimsical & Wonder", "Epic & Dramatic", "Fast-Paced Action",
    "Suspenseful & Noir", "Calm & Atmospheric", "Humorous & Comic", "Melancholic & Poetic",
]


# --- Style lock table -------------------------------------------------------
# Every segment prompt ends with the exact same style clause so the look cannot
# drift between shots. Keys must match VISUAL_STYLES.
STYLE_LOCKS = {
    "Cinematic Live-Action": "shot on 35mm anamorphic, shallow depth of field, natural motion blur, filmic contrast with clean highlight rolloff",
    "Anime / Studio Ghibli": "hand-painted 2D anime cel style, soft watercolour backgrounds, thick clean linework, gentle bloom",
    "3D Pixar / Disney CGI": "polished 3D CGI feature-animation look, soft global illumination, subsurface skin scattering, expressive stylised proportions",
    "Dark Fantasy / Gritty Gothic": "desaturated gothic palette, heavy chiaroscuro, volumetric haze, coarse film grain",
    "Cyberpunk Neon Noir": "neon-soaked night palette of magenta and cyan, wet reflective surfaces, hard practical lights, anamorphic flares",
    "80s Retro Sci-Fi Synthwave": "1980s optical-print look, magenta and teal gradients, halation around light sources, subtle gate weave",
    "Watercolor Storybook": "hand-painted watercolour illustration, visible paper grain, soft bleeding edges, warm muted pigments",
    "Vintage 70s Panavision Film": "1970s Panavision spherical look, warm amber grade, soft focus falloff, visible halation and grain",
    "Claymation / Stop-Motion": "handcrafted plasticine stop-motion, visible fingerprint texture, practical miniature set lighting, subtle frame-to-frame jitter",
    "Graphic Novel / High-Contrast Comic": "inked graphic-novel look, bold black hatching, flat limited palette, high-contrast rim light",
}

# --- Mood table -------------------------------------------------------------
MOOD_LOCKS = {
    "Whimsical & Wonder": ("warm golden key light with soft playful bounce", "buoyant and curious"),
    "Epic & Dramatic": ("hard directional key with deep contrasting shadows", "sweeping and momentous"),
    "Fast-Paced Action": ("crisp high-contrast daylight with hard edges", "urgent and kinetic"),
    "Suspenseful & Noir": ("low-key sidelight, pooled shadow, single practical source", "tense and withholding"),
    "Calm & Atmospheric": ("diffuse overcast light with gentle falloff", "still and contemplative"),
    "Humorous & Comic": ("bright even light with saturated colour", "light and mischievous"),
    "Melancholic & Poetic": ("pale window light with cool desaturated shadow", "wistful and unhurried"),
}

# --- Shot recipes -----------------------------------------------------------
# Each recipe is (opening shot size, opening camera move, cut shot size, cut
# camera move). The director walks this list with a seeded shuffle so no two
# consecutive segments open on the same framing or the same move.
SHOT_RECIPES = [
    ("a wide establishing shot", "the camera cranes up with large amplitude at slow speed over",
     "a medium close-up", "pushes in with small amplitude at medium speed on"),
    ("a low-angle medium shot", "the camera tracks forward with large amplitude at fast speed alongside",
     "an extreme close-up", "whip pans right with large amplitude at fast speed to"),
    ("an over-the-shoulder medium shot", "the camera orbits left with medium amplitude at slow speed around",
     "a wide shot", "pulls back with large amplitude at medium speed from"),
    ("a top-down wide shot", "the camera tilts down with medium amplitude at slow speed onto",
     "a medium wide shot", "cranes down with medium amplitude at medium speed to"),
    ("an extreme wide landscape shot", "the camera pans right with large amplitude at slow speed across",
     "a close-up", "pushes in with medium amplitude at fast speed on"),
    ("a handheld medium wide shot", "the camera follows handheld with medium amplitude at fast speed behind",
     "a low-angle close-up", "tilts up with small amplitude at medium speed to"),
    ("a static symmetrical wide shot", "the camera holds static on",
     "a medium shot", "tracks left with medium amplitude at medium speed with"),
    ("a close-up insert", "the camera pulls back with medium amplitude at medium speed from",
     "an extreme wide shot", "cranes up with large amplitude at slow speed above"),
    ("a dutch-angle medium shot", "the camera orbits right with large amplitude at medium speed around",
     "a medium close-up profile", "settles static on"),
    ("a silhouetted backlit wide shot", "the camera tracks backward with large amplitude at medium speed ahead of",
     "a detail insert", "tilts down with small amplitude at slow speed onto"),
]


# Titles whose full stop does not end a sentence, so "Dr. Maya Vance: ..." stays
# one subject instead of splitting into two.
TITLE_ABBREVIATIONS = ("Dr", "Mr", "Mrs", "Ms", "Prof", "Capt", "Sgt", "Lt", "St", "Jr", "Sr")


def _style_clause(effective_style, pacing_mood):
    """The single style sentence repeated verbatim at the end of every segment."""
    look = STYLE_LOCKS.get(effective_style)
    if look is None:
        look = f"{effective_style} look, held identical in every shot"
    light, _tone = MOOD_LOCKS.get(pacing_mood, ("naturalistic key light", "steady"))
    return (f"Style: {effective_style} - {look}; {light}. "
            "The same palette, grade, wardrobe and lighting logic hold for every shot of the story. "
            "No dissolves, no on-screen text, no subtitle bars, no watermarks.")


def _subject_lock(characters):
    """The character-bible line repeated verbatim in every segment prompt.

    MiniMax H3 has no memory between segments, so a bare <Subject 1> in segment 3
    renders a different person. Restating the description every time is what
    actually holds the cast together."""
    parts = []
    for c in characters:
        label = c.get("label", "<Subject 1>")
        desc = (c.get("appearance") or "").strip().rstrip(".")
        voice = (c.get("voice") or "").strip().rstrip(".")
        sid = c.get("id", "S1")
        line = f"{label} is {desc}"
        if voice and voice != "as described above":
            line += f"; voice ({sid}) is {voice}"
        elif voice:
            line += f"; speaks as ({sid})"
        parts.append(line + ".")
    return " ".join(parts)


def _parse_character_hints(character_hints, fallback_desc, fallback_voice):
    """Turns free-form 'Name: description' hint lines into tagged subjects."""
    chars = []
    raw = (character_hints or "").strip()
    if raw:
        chunks = []
        for block in re.split(r"[\n;]+", raw):
            block = block.strip()
            if not block:
                continue
            # "Maya: ... . Sprocket: ..." on one line is still two subjects, so split
            # ahead of any capitalised name that introduces its own description -
            # without being fooled by the full stop in "Dr. Maya Vance:".
            guarded = block
            for abbrev in TITLE_ABBREVIATIONS:
                guarded = guarded.replace(abbrev + ".", abbrev + "\x00")
            chunks.extend(c.strip().replace("\x00", ".") for c in
                          re.split(r"(?<=\.)\s+(?=[A-Z][\w .'-]{0,30}:)", guarded) if c.strip())
        for i, chunk in enumerate(chunks[:4]):
            if ":" in chunk:
                name, desc = chunk.split(":", 1)
            else:
                name, desc = f"Subject {i+1}", chunk
            has_voice_note = re.search(r"\bvoice|accent|speaks|tone\b", desc, re.IGNORECASE)
            chars.append({
                "id": f"S{i+1}",
                "label": f"<Subject {i+1}>",
                "name": name.strip(),
                "appearance": f"{name.strip()}, {desc.strip()}",
                # Only the lead gets a default voice; extra subjects stay silent unless
                # the hints describe how they sound.
                "voice": fallback_voice if i == 0 else ("as described above" if has_voice_note else ""),
            })
    if not chars:
        chars = [{"id": "S1", "label": "<Subject 1>", "name": "The Protagonist",
                  "appearance": fallback_desc, "voice": fallback_voice}]
    return chars


def _format_timecode(seconds):
    seconds = max(0.1, float(seconds))
    return "00:%04.1f" % seconds if seconds < 60 else "%02d:%04.1f" % (seconds // 60, seconds % 60)


_DIALOGUE_RE = re.compile(r"<d>\s*(?!\[)", re.IGNORECASE)


def _normalize_shot_text(desc, cut_time):
    """Repairs the structural rules an LLM most often drops."""
    text = " ".join((desc or "").split())
    text = re.sub(r"^integrated_multimodal_description\s*:\s*", "", text, flags=re.IGNORECASE)

    # Second-person / instruction leakage is never part of a shot description.
    text = re.sub(r"\b(make sure to|be sure to|you should|please)\b\s*", "", text, flags=re.IGNORECASE)

    if "[Shot 1]" not in text:
        # A model that opens straight on "[Shot 2]" means that shot to BE the opener;
        # prefixing "[Shot 1]" there would produce an empty first shot.
        if text.lstrip().startswith("[Shot 2]"):
            text = text.replace("[Shot 2]", "[Shot 1]", 1)
            text = re.sub(r"^\[Shot 1\]\s*At\s*\d{1,2}:\d{2}(\.\d+)?\s*,?\s*", "[Shot 1] ", text)
        else:
            text = "[Shot 1] " + text

    # A [Shot 2] with no timestamp desyncs the cut from the audio.
    if "[Shot 2]" in text and not re.search(r"\[Shot 2\]\s*At\s*\d", text):
        text = re.sub(r"\[Shot 2\]\s*", f"[Shot 2] At {_format_timecode(cut_time)}, ", text)

    # Every spoken line needs an explicit language tag.
    text = _DIALOGUE_RE.sub("<d>[English] ", text)
    return text.strip()


def _compose_segment_prompt(desc, soundscape, music, subject_lock, style_clause, cut_time):
    """Assembles the final MiniMax H3 prompt block in base-en.txt field order."""
    shots = _normalize_shot_text(desc, cut_time)
    body = f"{subject_lock} {shots} {style_clause}"
    return (f"integrated_multimodal_description: {body}\n\n"
            f"overall_soundscape: {soundscape}\n\n"
            f"non_diegetic_music: {music}")


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
CUSTOM_MODEL_CHOICE = "custom (use ollama_model_custom)"

# Offered in the dropdown even when they are not pulled yet. Ordered by how well they
# hold the MiniMax H3 prompt format, best first; the runtime falls back down this list
# when the chosen model is missing and auto-pull is off.
OLLAMA_RECOMMENDED = [
    "qwen3:8b",          # default: best quality-per-second here, and only 5GB of VRAM
    "qwen3:30b",         # highest measured score, but ~7x slower and 18GB resident
    "qwen3:14b",
    "qwen3:4b",          # for smaller cards
    "qwen3:30b-a3b",     # MoE, fast for its size
    "llama3.1:8b",
    "gemma3:12b",
    "mistral-nemo:12b",
    "qwen2.5:7b",        # previous default, kept so old workflows still load
]

# Substrings of model names that cannot answer a prompt, so they are never picked
# as an automatic fallback (an embedding model would fail silently and confusingly).
NON_CHAT_HINTS = ("embed", "embedding", "reranker", "bge-", "-base")

_TAGS_CACHE = {"url": None, "models": [], "stamp": 0.0}
_TAGS_TTL = 20.0        # seconds; INPUT_TYPES is called often, Ollama should not be


def _ollama_tags(url=DEFAULT_OLLAMA_URL, timeout=1.5, use_cache=True):
    """The models this Ollama actually has. Empty list when it is not reachable."""
    now = time.time()
    if use_cache and _TAGS_CACHE["url"] == url and now - _TAGS_CACHE["stamp"] < _TAGS_TTL:
        return list(_TAGS_CACHE["models"])
    models = []
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        pass                                  # not running, wrong port, no network - fine
    _TAGS_CACHE.update({"url": url, "models": models, "stamp": now})
    return list(models)


def _is_chat_model(name):
    lowered = name.lower()
    return not any(hint in lowered for hint in NON_CHAT_HINTS)


def ollama_model_options():
    """Dropdown contents: installed models first, then recommendations you could pull.

    The entries are plain model names and nothing else. It is tempting to tag the ones
    that are not downloaded, but the tag would become part of the saved value, and the
    same workflow opened on a machine with a different set of models installed would
    show an invalid widget. Which models are present is reported in the tooltip and on
    the console instead."""
    installed = [m for m in _ollama_tags() if _is_chat_model(m)]
    options = list(installed)
    for name in OLLAMA_RECOMMENDED:
        if name not in options:
            options.append(name)
    # A fresh node should start on the default.
    if DEFAULT_OLLAMA_MODEL in options:
        options.insert(0, options.pop(options.index(DEFAULT_OLLAMA_MODEL)))
    else:
        options.insert(0, DEFAULT_OLLAMA_MODEL)
    options.append(CUSTOM_MODEL_CHOICE)
    return options


def ollama_model_tooltip():
    """Says which of the offered models are actually on this machine."""
    installed = [m for m in _ollama_tags() if _is_chat_model(m)]
    if not installed:
        return ("No Ollama answered at " + DEFAULT_OLLAMA_URL + ". The node still runs - it "
                "falls back to its built-in procedural writer. Start Ollama and refresh the "
                "browser to see your models here.")
    missing = [m for m in OLLAMA_RECOMMENDED if m not in installed]
    text = "Installed on this machine: " + ", ".join(installed) + "."
    if missing:
        text += (" Offered but not downloaded: " + ", ".join(missing) +
                 " - these are pulled automatically on first use when auto_pull_model is on, "
                 "or run 'ollama pull <name>' yourself.")
    return text


def _clean_model_name(choice):
    return (choice or "").split("  (")[0].strip()


def _pull_ollama_model(url, model, timeout=1800):
    """Downloads a model through Ollama. Returns True when it is available afterwards."""
    print(f"[Geekatplay Studio] Model '{model}' is not on this Ollama yet. Downloading it "
          f"now - this happens once and can take several minutes...")
    try:
        payload = json.dumps({"model": model, "stream": False}).encode("utf-8")
        req = urllib.request.Request(f"{url.rstrip('/')}/api/pull", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if str(result.get("status", "")).lower() in ("success", "ok"):
            print(f"[Geekatplay Studio] Downloaded '{model}'.")
            _TAGS_CACHE["stamp"] = 0.0
            return True
        print(f"[Geekatplay Studio] Ollama did not confirm the download of '{model}': {result}")
    except Exception as exc:
        print(f"[Geekatplay Studio] Could not download '{model}' ({exc}).")
    return False


def free_ollama_vram(url=DEFAULT_OLLAMA_URL, timeout=10):
    """Asks Ollama to drop every loaded model, freeing the card for the renderer.

    The prompt-writing nodes deliberately keep their model resident so a chain of them
    does not reload it for every segment. That leaves the weights in VRAM when the
    render begins, which on a single card is exactly where they must not be. A renderer
    calls this once, just before it loads the diffusion models."""
    freed = []
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/ps", timeout=timeout) as resp:
            running = json.loads(resp.read().decode("utf-8")).get("models", [])
    except Exception:
        return freed                     # no Ollama here, nothing to free

    for entry in running:
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        try:
            payload = json.dumps({"model": name, "prompt": "", "keep_alive": 0}).encode()
            req = urllib.request.Request(f"{url.rstrip('/')}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=timeout).read()
            freed.append((name, entry.get("size_vram", 0)))
        except Exception as exc:
            print(f"[Geekatplay Studio] Could not unload '{name}' from Ollama ({exc}).")

    if freed:
        total = sum(v for _n, v in freed) / 1e9
        print(f"[Geekatplay Studio] Freed {total:.1f} GB of VRAM from Ollama before rendering "
              f"({', '.join(n for n, _v in freed)}).")
    return freed


def resolve_ollama_model(url, choice, custom="", auto_pull=True):
    """Works out which model to actually query.

    Order: the custom name if one is typed, then the chosen model if it is installed,
    then a download if auto_pull is on, then the best installed model from the
    recommended list, then whatever else is installed. Returns None when Ollama has
    nothing at all - the caller then uses its procedural fallback."""
    wanted = (custom or "").strip() or _clean_model_name(choice)
    if not wanted or wanted == _clean_model_name(CUSTOM_MODEL_CHOICE):
        wanted = DEFAULT_OLLAMA_MODEL

    installed = _ollama_tags(url, timeout=3.0, use_cache=False)
    if not installed:
        print(f"[Geekatplay Studio] No Ollama at {url} (or it has no models).")
        return None

    # Ollama reports "qwen3:8b"; accept a bare "qwen3" as meaning its default tag.
    if wanted in installed:
        return wanted
    bare = [m for m in installed if m.split(":")[0] == wanted.split(":")[0]]
    if ":" not in wanted and bare:
        print(f"[Geekatplay Studio] Using installed '{bare[0]}' for '{wanted}'.")
        return bare[0]

    if auto_pull and _pull_ollama_model(url, wanted):
        return wanted

    for candidate in OLLAMA_RECOMMENDED:
        if candidate in installed:
            print(f"[Geekatplay Studio] '{wanted}' is not installed; falling back to "
                  f"'{candidate}'. Turn on auto_pull_model to download '{wanted}' instead, "
                  f"or run: ollama pull {wanted}")
            return candidate
    usable = [m for m in installed if _is_chat_model(m)]
    if not usable:
        print(f"[Geekatplay Studio] Ollama has no model that can write text "
              f"(found: {', '.join(installed)}). Run: ollama pull {wanted}")
        return None
    print(f"[Geekatplay Studio] '{wanted}' is not installed; falling back to the first "
          f"installed model '{usable[0]}'. To get '{wanted}': ollama pull {wanted}")
    return usable[0]


def align_frame_count(duration_seconds: float, fps: int = FPS) -> int:
    """MiniMax H3 requires frames to follow the 17k + 5 grid (17 frames per latent block)."""
    raw_frames = max(5, int(round(duration_seconds * fps)))
    k = max(0, round((raw_frames - 5) / 17))
    return 17 * k + 5


# Screen size, separate from screen shape. The aspect ratio decides the proportions;
# this decides how many pixels they are rendered at. MiniMax H3's canvas caps the long
# edge at 1344 and every axis must be a multiple of 32.
FRAME_SIZES = {
    "auto (MiniMax H3 native, ~1 MP)": None,
    "1344 long edge (H3 maximum)": 1344,
    "1152 long edge": 1152,
    "1024 long edge": 1024,
    "896 long edge": 896,
    "768 long edge": 768,
    "640 long edge (fast preview)": 640,
    "custom (width x height below)": "custom",
}
CANVAS_CAP = 1344
# MiniMax H3's canvas rule is a 768 short edge with a 768x1344 area cap. Going past it
# is not just slow, it is outside what the model was trained on.
CANVAS_MAX_PIXELS = 768 * 1344


def snap_axis(value, multiple=32, cap=CANVAS_CAP):
    return int(min(cap, max(multiple, round(value / multiple) * multiple)))


def canvas_from_ratio(ar_w, ar_h, long_edge=None, megapixels=0.98, multiple=32):
    """Width and height for an aspect ratio, either at a given long edge or at an area."""
    ar_w, ar_h = float(ar_w), float(ar_h)
    if long_edge:
        if ar_w >= ar_h:
            width, height = long_edge, long_edge * ar_h / ar_w
        else:
            width, height = long_edge * ar_w / ar_h, long_edge
    else:
        total = megapixels * 1024 * 1024
        width = math.sqrt(total * ar_w / ar_h)
        height = total / width
    if width * height > CANVAS_MAX_PIXELS:
        scale = math.sqrt(CANVAS_MAX_PIXELS / (width * height))
        width, height = width * scale, height * scale
    return snap_axis(width, multiple), snap_axis(height, multiple)


def resolve_canvas(aspect_choice, frame_size="auto (MiniMax H3 native, ~1 MP)",
                   custom_width=1344, custom_height=768, image=None):
    """The one place canvas size is decided, shared by the director and the manager.

    'match first keyframe' takes the shape from the supplied image; everything else
    takes it from the aspect dropdown. Either way `frame_size` decides the pixels."""
    size = FRAME_SIZES.get(frame_size, None)
    if size == "custom":
        width, height = snap_axis(custom_width), snap_axis(custom_height)
        if width * height > CANVAS_MAX_PIXELS:
            scale = math.sqrt(CANVAS_MAX_PIXELS / (width * height))
            capped = snap_axis(width * scale), snap_axis(height * scale)
            print(f"[Geekatplay Studio] {width}x{height} is past MiniMax H3's canvas limit; "
                  f"rendering at {capped[0]}x{capped[1]} instead.")
            return capped
        return width, height

    if aspect_choice == "match first keyframe" and image is not None:
        ar_w, ar_h = int(image.shape[2]), int(image.shape[1])
        if size is None:
            # No explicit size: use H3's own canvas rule for this image shape.
            return h3_adapt_canvas(ar_w, ar_h)
    else:
        ar_w, ar_h = ASPECT_RATIOS.get(aspect_choice, (16, 9))

    return canvas_from_ratio(ar_w, ar_h, long_edge=size)


def h3_adapt_canvas(width, height):
    """MiniMax H3's own canvas rule, imported lazily so this module stays importable
    outside a running ComfyUI."""
    from comfy_extras import nodes_minimax_h3 as _h3
    return _h3.adapt_canvas(width, height)


def calculate_resolution(aspect_name: str, megapixels: float = 0.98, multiple: int = 32):
    """Calculates width and height snapped to 32px boundary, capped at 1344 on long edge."""
    ar_w, ar_h = ASPECT_RATIOS.get(aspect_name, (16, 9))
    total_pixels = megapixels * 1024 * 1024
    width = math.sqrt(total_pixels * ar_w / ar_h)
    height = total_pixels / width
    width = min(1344, max(multiple, round(width / multiple) * multiple))
    height = min(1344, max(multiple, round(height / multiple) * multiple))
    return int(width), int(height)


def _load_reference_guides():
    """Dynamically reads the two official prompt reference files from workflows/"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wf_dir = os.path.join(base_dir, "workflows")
    base_path = os.path.join(wf_dir, "base-en.txt")
    ref_path = os.path.join(wf_dir, "ref-en.txt")

    base_guide = ""
    ref_guide = ""

    if os.path.exists(base_path):
        try:
            with open(base_path, "r", encoding="utf-8", errors="replace") as f:
                base_guide = f.read().strip()
        except Exception as e:
            print(f"[Geekatplay Studio] Warning: could not load base-en.txt: {e}")

    if os.path.exists(ref_path):
        try:
            with open(ref_path, "r", encoding="utf-8", errors="replace") as f:
                ref_guide = f.read().strip()
        except Exception as e:
            print(f"[Geekatplay Studio] Warning: could not load ref-en.txt: {e}")

    return base_guide, ref_guide


def _call_ollama(url: str, model: str, prompt: str, timeout: int = 300,
                 seed: int = 42, temperature: float = 0.95, keep_alive=0) -> str:
    """Sends a generation prompt to a local Ollama server.

    The seed is forwarded so a given node seed reproduces the same storyboard, and the
    temperature runs hot enough to get varied staging out of the model.

    Reasoning models such as Qwen3 otherwise spend their output budget thinking before
    answering, so thinking is switched off. Older Ollama builds reject that field, so
    the call is retried without it.

    keep_alive=0 makes Ollama drop the model from VRAM the moment it answers. That
    matters here: the director runs immediately before a MiniMax H3 render on the same
    card, and Ollama's default is to hold the weights for five minutes. On a 24GB card
    a resident 18GB model and a FastH3 render do not both fit."""
    endpoint = f"{url.rstrip('/')}/api/generate"
    options = {
        "temperature": temperature,
        "top_p": 0.92,
        "repeat_penalty": 1.15,
        "seed": int(seed) % (2 ** 31),
        "num_ctx": 16384,
    }

    def _post(include_think):
        payload = {"model": model, "prompt": prompt, "stream": False,
                   "format": "json", "keep_alive": keep_alive, "options": options}
        if include_think:
            payload["think"] = False
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "")

    try:
        return _post(True)
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 422):
            raise
        return _post(False)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json_block(text: str) -> dict:
    """Robust JSON extraction from an LLM response, markdown or raw.

    A reasoning model's <think> block is stripped first: it often contains braces and
    draft JSON, which would otherwise be picked up instead of the real answer."""
    text = _THINK_RE.sub("", text or "")
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)  # unclosed
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    raise ValueError("No valid JSON found in LLM response.")


ENVIRONMENTS = {
    "beach": ("a sunlit coastline of golden sand and azure surf",
              "Waves break and hiss across wet sand, coastal wind combs the dunes, distant gulls call.",
              "Breezy acoustic guitar with marimba accents, building gently."),
    "ocean": ("open ocean under rolling turquoise swell",
              "Deep rhythmic swell, hull creak, wind across open water.",
              "Sustained strings and harp at a lyrical tempo."),
    "forest": ("an ancient forest dappled with emerald sunbeams and mossy stone",
               "Canopy leaves rustle, twigs snap underfoot, songbirds trade calls.",
               "Warm woodwinds and strings at a serene tempo."),
    "city": ("a rain-slicked urban street of reflective puddles and neon signage",
             "Traffic hum, a distant train rumble, water splashing underfoot.",
             "Pulsing synth bass under atmospheric electric piano."),
    "snow": ("a frozen landscape blanketed in deep glistening snow",
             "Icy wind whistles over drifts, boots compress dry powder.",
             "Bowed cello and shimmering high bells at a slow tempo."),
    "desert": ("windswept red sand under an expansive twilight sky",
               "Wind drives fine dust across sandstone, grit ticks against rock.",
               "Resonant slide guitar and frame drums at a steady tempo."),
    "greenhouse": ("a sunlit high-tech greenhouse of misted glass and terraced planters",
                   "Misters hiss in cycles, glass panes tick as they warm, leaves brush together.",
                   "Glassy synth pads over a slow arpeggiated pulse."),
    "meadow": ("a wildflower meadow bathed in warm golden-hour light",
               "Summer breeze combs prairie grass, crickets saw steadily.",
               "Soaring violin over rhythmic nylon-string guitar."),
    "cave": ("a vaulted limestone cave streaked with mineral colour",
             "Water drips into a still pool, footfalls slap and echo off wet stone.",
             "Low drones and sparse struck metal at a patient tempo."),
    "temple": ("a half-buried stone temple choked with vines and drifting dust",
               "Dust sifts from cracked lintels, stone grinds faintly against stone.",
               "Low choir and struck bowls swelling slowly."),
}

# Verb pools used to keep multi-segment stories from repeating an action.
ACTION_BEATS = [
    "moves into the space, scanning it with wary attention",
    "stops short, caught by something just out of reach",
    "reaches out and makes contact for the first time",
    "recoils as the situation turns, recovering in the same breath",
    "presses forward against resistance, committed now",
    "turns to look back at the distance already covered",
    "works quickly with focused hands, solving the problem in motion",
    "holds still while the environment changes around them",
    "breaks into motion, closing the distance fast",
    "settles, and takes in what has changed",
]


def _dynamic_contextual_director(story_premise: str, character_hints: str, visual_style: str,
                                 pacing_mood: str, num_segments: int, segment_duration: float,
                                 seed: int) -> dict:
    """Procedural fallback used when Ollama is unavailable.

    Seeded so a given seed always yields the same storyboard, but no two segments
    reuse a framing, a camera move or an action beat."""
    rng = random.Random(seed)

    premise = story_premise.strip() or "A loyal companion on an unexpected journey."
    p_lower = premise.lower()
    c_lower = (character_hints or "").lower()

    is_dog = any(w in p_lower or w in c_lower for w in
                 ["dog", "puppy", "pup", "hound", "retriever", "corgi", "husky", "terrier", "canine"])
    is_cat = any(w in p_lower or w in c_lower for w in ["cat", "kitten", "kitty", "feline"])

    env_name, (env_desc, env_sound, env_music) = "cinematic setting", (
        "a scenic location bathed in rich natural light",
        "Ambient air movement with sparse distant detail.",
        "Warm strings and gentle piano at a steady tempo.")
    # "a greenhouse at the edge of the desert" is a greenhouse story: the earliest
    # keyword in the premise wins, rather than whichever key the dict lists first.
    hits = [(p_lower.index(k), k) for k in ENVIRONMENTS if k in p_lower]
    hits += [(10_000 + c_lower.index(k), k) for k in ENVIRONMENTS if k in c_lower and k not in p_lower]
    if hits:
        _pos, k = min(hits)
        env_name, (env_desc, env_sound, env_music) = k, ENVIRONMENTS[k]

    if is_dog:
        default_desc = ("an energetic golden-furred retriever with bright expressive eyes, "
                        "floppy ears and a worn red collar")
        default_voice = "bright eager barks and close panting"
        foley_subject = "Paws patter and skid, a collar tag jingles, breath comes in quick pants"
    elif is_cat:
        default_desc = ("a sleek calico cat with luminous green eyes, white paws and "
                        "constantly reading whiskers")
        default_voice = "soft inquisitive chirps and a low steady purr"
        foley_subject = "Padded paws land almost silently, claws tick once on hard surface, a purr rolls underneath"
    else:
        default_desc = ("a focused traveller in worn practical clothing with keen observant eyes")
        default_voice = "a clear, measured voice with a warm low register"
        foley_subject = "Footsteps land deliberately, fabric shifts with each movement, breath stays controlled"

    characters = _parse_character_hints(character_hints, default_desc, default_voice)
    subject_lock = _subject_lock(characters)
    style_clause = _style_clause(visual_style, pacing_mood)
    lead = characters[0]

    recipes = SHOT_RECIPES[:]
    rng.shuffle(recipes)
    beats = ACTION_BEATS[:]
    rng.shuffle(beats)

    lines = [
        "<d>[English] There it is.</d>",
        "<d>[English] That should not be here.</d>",
        "<d>[English] Hold still.</d>",
        "<d>[English] One more step.</d>",
        "<d>[English] I know that sound.</d>",
        "<d>[English] Almost.</d>",
    ]
    rng.shuffle(lines)

    music_moves = [
        "It enters quietly under the action and holds.",
        "It swells through the middle of the shot and settles.",
        "It thins to a single sustained line, leaving room for the Foley.",
        "It drops out entirely on the cut, then returns underneath.",
        "It builds steadily and lands hard on the final beat.",
    ]
    rng.shuffle(music_moves)

    air = [
        "The air is still enough to hear the room itself.",
        "A gust crosses the frame and dies away.",
        "Something far off answers, once.",
        "Everything close is sharp; everything distant is soft.",
    ]
    rng.shuffle(air)

    cut_time = round(segment_duration * 0.52, 1)
    segments = []
    for i in range(num_segments):
        open_size, open_move, cut_size, cut_move = recipes[i % len(recipes)]
        beat = beats[i % len(beats)]
        tag = lead["label"]

        support = ""
        if len(characters) > 1:
            other = characters[1 + (i % (len(characters) - 1))]
            support = (f" {other['label']} holds the {'foreground' if i % 2 else 'background'} "
                       f"of the frame, reacting to the same moment.")

        camera = open_move[0].upper() + open_move[1:]
        shot_1 = (f"[Shot 1] {open_size} of {tag}, who {beat}, in {env_desc}.{support} "
                  f"{camera} {tag}.")
        if is_dog or is_cat:
            voice_line = f"{tag} sounds off - {lead['voice']}."
        else:
            voice_line = f"{tag} ({lead['id']}) says: {lines[i % len(lines)]}"
        shot_2 = (f"[Shot 2] At {_format_timecode(cut_time)}, the camera cuts to {cut_size} "
                  f"and {cut_move} {tag}. {voice_line}")

        foley = f"{foley_subject}. {env_sound} {air[i % len(air)]}"
        music = f"{env_music} {music_moves[i % len(music_moves)]}"
        prompt = _compose_segment_prompt(f"{shot_1} {shot_2}", foley, music,
                                         subject_lock, style_clause, cut_time)

        segments.append({
            "segment_index": i + 1,
            "duration": float(segment_duration),
            "frame_count": align_frame_count(segment_duration),
            "summary": f"Segment {i+1}: {open_size} - {lead['name']} {beat}",
            "prompt": prompt,
            "soundscape": foley,
            "music": music,
        })

    bible_lines = []
    for c in characters:
        line = f"- {c['label']} ({c['id']}): {c['appearance']}."
        if c.get("voice"):
            line += f" Voice: {c['voice']}."
        bible_lines.append(line)
    bible_text = "\n".join(bible_lines + [f"Environment: {env_desc}", f"Style lock: {style_clause}"])

    return {
        "story_title": f"Story: {premise[:40]}",
        "story_premise": premise,
        "visual_style": visual_style,
        "pacing_mood": pacing_mood,
        "character_bible": bible_text,
        "characters": characters,
        "subject_lock": subject_lock,
        "style_clause": style_clause,
        "segments": segments,
    }


class GAPStoryDirector:
    """Geekatplay Studio - Vladimir Chopine.
    Story Director node for MiniMax FastH3 Lite.
    Loads the official prompt reference guides (base-en.txt and ref-en.txt) and passes
    them directly to local Ollama. The AI formulates a completely custom, dynamic
    cinematic script with character consistency, rich camera motion, cuts, soundscapes,
    and dialogue strictly complying with MiniMax H3 standards."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_premise": ("STRING", {
                    "multiline": True,
                    "default": "A joyful golden retriever puppy finds a lost vintage pocket watch buried in the sand on the beach and brings it back to its owner.",
                    "tooltip": "The core story idea, plot, or narrative premise."
                }),
                "visual_style": (VISUAL_STYLES, {
                    "default": "Cinematic Live-Action",
                    "tooltip": "Primary visual aesthetic and rendering style."
                }),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {
                    "default": "16:9 (Landscape)",
                    "tooltip": "Screen SHAPE. The proportions of the frame."
                }),
                "frame_size": (list(FRAME_SIZES.keys()), {
                    "default": "auto (MiniMax H3 native, ~1 MP)",
                    "tooltip": "Screen SIZE. How many pixels the shape is rendered at. Smaller is "
                               "much faster to render; 1344 is the most MiniMax H3 accepts on the "
                               "long edge. Both axes snap to 32."
                }),
                "pacing_and_mood": (PACING_MOODS, {
                    "default": "Whimsical & Wonder",
                    "tooltip": "Tone, tempo, and emotional feeling."
                }),
                "num_segments": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "How many sequential video segments to create."
                }),
                "segment_duration": ("FLOAT", {
                    "default": 5.0, "min": 2.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Target duration per segment in seconds. Automatically snaps to MiniMax H3's 17k+5 frame grid at 24fps."
                }),
            },
            "optional": {
                "character_hints": ("STRING", {
                    "multiline": True,
                    "default": "Cooper: 6-month-old golden retriever with shiny gold coat and red collar. Sarah: friendly young woman in summer dress.",
                    "tooltip": "Names, appearance details, clothing, accessories, or voice traits for characters."
                }),
                "custom_width": ("INT", {
                    "default": 1344, "min": 32, "max": 1344, "step": 32,
                    "tooltip": "Used only when frame_size is 'custom'."
                }),
                "custom_height": ("INT", {
                    "default": 768, "min": 32, "max": 1344, "step": 32,
                    "tooltip": "Used only when frame_size is 'custom'."
                }),
                "custom_style": ("STRING", {
                    "default": "",
                    "tooltip": "Custom art style if 'Custom' is selected above."
                }),
                "use_ollama": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Query local Ollama LLM with base-en.txt and ref-en.txt reference guides. Falls back dynamically if offline."
                }),
                "ollama_url": ("STRING", {
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "URL of local Ollama server."
                }),
                "ollama_model": (ollama_model_options(), {
                    "default": DEFAULT_OLLAMA_MODEL,
                    "tooltip": "Models on your Ollama are listed first, then ones worth pulling. "
                               + ollama_model_tooltip()
                }),
                "ollama_model_custom": ("STRING", {
                    "default": "",
                    "tooltip": "Any model name, used instead of the dropdown when filled in. "
                               "For models that are not in the list."
                }),
                "auto_pull_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Download the chosen model through Ollama if it is missing. Off, "
                               "the node falls back to the best model you already have."
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Leave the LLM in VRAM between calls so a chain of story nodes "
                               "does not reload it every time. The renderer frees it once, right "
                               "before it loads the video models, so nothing is wasted. Turn off "
                               "to drop it the moment this node is done."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible story and camera composition."
                }),
            }
        }

    RETURN_TYPES = ("GAP_STORY_PLAN", "STRING", "STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("story_plan", "story_summary", "character_bible", "prompts_preview", "segment_count", "width", "height", "frame_length")
    FUNCTION = "generate_story"
    CATEGORY = "Geekatplay/StoryTeller"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate_story(self, story_premise, visual_style, aspect_ratio,
                       frame_size="auto (MiniMax H3 native, ~1 MP)",
                       pacing_and_mood="Whimsical & Wonder",
                       num_segments=3, segment_duration=5.0, character_hints="",
                       custom_width=1344, custom_height=768, custom_style="",
                       use_ollama=True, ollama_url=DEFAULT_OLLAMA_URL,
                       ollama_model=DEFAULT_OLLAMA_MODEL, ollama_model_custom="",
                       auto_pull_model=True, keep_model_loaded=True, seed=42, **kwargs):
        pacing_mood = pacing_and_mood
        effective_style = custom_style.strip() if visual_style == "Custom" and custom_style.strip() else visual_style

        frame_length = align_frame_count(segment_duration, FPS)
        actual_duration = round(frame_length / FPS, 2)
        width, height = resolve_canvas(aspect_ratio, frame_size, custom_width, custom_height)

        story_dict = None

        # The cast and the look are decided here, once, and then forced onto every
        # segment regardless of whether Ollama or the fallback wrote the shots.
        rng = random.Random(seed)
        characters = _parse_character_hints(
            character_hints,
            "a focused traveller in worn practical clothing with keen observant eyes",
            "a clear, measured voice with a warm low register")
        subject_lock = _subject_lock(characters)
        style_clause = _style_clause(effective_style, pacing_mood)
        cut_time = round(actual_duration * 0.52, 1)

        recipes = SHOT_RECIPES[:]
        rng.shuffle(recipes)
        shot_plan = "\n".join(
            f"  Segment {i+1}: open on {recipes[i % len(recipes)][0]}, "
            f"{recipes[i % len(recipes)][1].replace('the camera ', '')} the subject; "
            f"cut to {recipes[i % len(recipes)][2]}, "
            f"{recipes[i % len(recipes)][3]} the subject."
            for i in range(num_segments))

        if use_ollama:
            resolved_model = resolve_ollama_model(
                ollama_url, ollama_model, ollama_model_custom, auto_pull_model)
            if resolved_model is None:
                use_ollama = False

        if use_ollama:
            base_guide, ref_guide = _load_reference_guides()

            system_instruction = f"""You are the Master Story Director for MiniMax H3 / FastH3 video generation.
Your job is to transform a user's story premise into a cohesive {num_segments}-segment cinematic storyboard.

=== REFERENCE GUIDE 1: BASE PROMPTING GUIDE (base-en.txt) ===
{base_guide}

=== REFERENCE GUIDE 2: FULL-REFERENCE MODE OUTPUT GUIDE (ref-en.txt) ===
{ref_guide}

THE CAST (verbatim, do not rename, restyle or re-age anyone):
{subject_lock}

THE LOOK (every segment is this look, no exceptions):
{style_clause}

PER-SEGMENT SHOT PLAN - follow it exactly, it exists so no two segments look alike:
{shot_plan}

CRITICAL RULES TO ENFORCE:
1. "integrated_multimodal_description" per segment:
   - "[Shot 1] " opens with the framing from the shot plan, then the subject and the action,
     then the environment, then the camera move, then the lighting.
   - The camera move is written with the official vocabulary: motion type + amplitude
     (small / medium / large) + speed (slow / medium / fast), e.g. "the camera tracks forward
     with large amplitude at fast speed alongside <Subject 1>".
   - A timed mid-shot cut: "[Shot 2] At {_format_timecode(cut_time)}, the camera cuts to ..." using the cut framing
     from the shot plan.
   - Refer to the cast only as <Subject 1>, <Subject 2>. Speech is tagged with the matching
     speaker id and wrapped: <Subject 1> (S1) says: <d>[English] line here.</d>
   - Keep spoken lines under 12 words. One speaker per shot.
2. CONSISTENCY - this is the thing that breaks most often:
   - Wardrobe, hair, colour, age, build and props are fixed by THE CAST above and never change.
   - The palette, grade and lighting logic are fixed by THE LOOK above and never change.
   - Carry the location, time of day and weather forward between segments unless the story
     deliberately relocates, and say so when it does.
   - End each segment on a stable pose and open the next one from that pose.
3. CREATIVITY - within those locks, push the staging:
   - Never repeat a camera move, a framing or a lighting idea from an earlier segment.
   - Use depth: put something in the foreground, midground and background of each shot.
   - Let the environment act - weather, dust, reflections, light changing through the shot.
   - Give the subject one specific physical detail per shot (what the hands do, where the
     eyes go, what the weight is on) instead of a generic description.
   - Vary the emotional register segment to segment; do not write the same beat twice.
4. Do not write instructions, second person, or "the video shows". Describe only what the
   camera sees and the microphone hears, in present tense.
5. "overall_soundscape": 1-4 sentences of physical Foley and ambience, tied to what is on screen.
6. "non_diegetic_music": 1-3 sentences, instruments + tempo + where it swells or drops out,
   continuous in character across all {num_segments} segments.

Respond ONLY with a valid JSON object matching this schema:
{{
  "story_title": "...",
  "character_bible": "...",
  "segments": [
    {{
      "segment_index": 1,
      "summary": "...",
      "integrated_multimodal_description": "...",
      "overall_soundscape": "...",
      "non_diegetic_music": "..."
    }}
  ]
}}"""

            _mood_light, _mood_tone = MOOD_LOCKS.get(pacing_mood, ("naturalistic key light", "steady"))
            user_prompt = f"""Story Premise: {story_premise}
Character Notes: {character_hints}
Visual Style: {effective_style}
Pacing & Mood: {pacing_mood} - the storytelling register is {_mood_tone}
Number of Segments: {num_segments}
Duration per Segment: {actual_duration}s ({frame_length} frames)
Canvas: {width}x{height}

Write a story arc with a beginning, a turn and a landing across the {num_segments} segments.
Each segment must advance the story - no segment may restate the previous one."""

            print(f"[Geekatplay Studio] Querying Ollama ({resolved_model}) with base-en.txt and ref-en.txt reference guides...")
            try:
                raw_response = _call_ollama(ollama_url, resolved_model,
                                            f"{system_instruction}\n\nUser Request:\n{user_prompt}\n\nJSON Response:",
                                            seed=seed,
                                            keep_alive="15m" if keep_model_loaded else 0)
                parsed = _extract_json_block(raw_response)

                if "segments" in parsed and len(parsed["segments"]) >= num_segments:
                    char_bible_raw = parsed.get("character_bible", "")
                    if isinstance(char_bible_raw, dict):
                        char_bible_str = "\n".join(f"- {k}: {v}" for k, v in char_bible_raw.items())
                    else:
                        char_bible_str = str(char_bible_raw)

                    if char_bible_str.strip():
                        char_bible_str = char_bible_str.strip() + "\n\nLocked cast: " + subject_lock
                    else:
                        char_bible_str = "Locked cast: " + subject_lock
                    char_bible_str += "\nStyle lock: " + style_clause

                    story_dict = {
                        "story_title": parsed.get("story_title", "Cinematic Story"),
                        "story_premise": story_premise,
                        "visual_style": effective_style,
                        "pacing_mood": pacing_mood,
                        "character_bible": char_bible_str,
                        "characters": characters,
                        "subject_lock": subject_lock,
                        "style_clause": style_clause,
                        "segments": []
                    }
                    for seg in parsed["segments"][:num_segments]:
                        desc = seg.get("integrated_multimodal_description", "")
                        sound = seg.get("overall_soundscape", "") or "Ambient room tone with sparse physical detail."
                        mus = seg.get("non_diegetic_music", "") or "Sparse sustained score, unobtrusive under the action."
                        # The locks are re-applied here: whatever the LLM wrote, every segment
                        # ships with the identical cast description and the identical style clause.
                        full_prompt = _compose_segment_prompt(
                            desc, sound, mus, subject_lock, style_clause, cut_time)
                        story_dict["segments"].append({
                            "segment_index": seg.get("segment_index", len(story_dict["segments"]) + 1),
                            "duration": actual_duration,
                            "frame_count": frame_length,
                            "summary": seg.get("summary", ""),
                            "prompt": full_prompt,
                            "soundscape": sound,
                            "music": mus,
                        })
                    print(f"[Geekatplay Studio] Ollama successfully formulated dynamic story '{story_dict['story_title']}'.")
            except Exception as e:
                print(f"[Geekatplay Studio] Ollama query failed ({e}). Falling back to dynamic contextual engine.")

        # Fallback to dynamic contextual engine if Ollama was not used or failed
        if story_dict is None:
            story_dict = _dynamic_contextual_director(
                story_premise=story_premise,
                character_hints=character_hints,
                visual_style=effective_style,
                pacing_mood=pacing_mood,
                num_segments=num_segments,
                segment_duration=actual_duration,
                seed=seed
            )
            print("[Geekatplay Studio] Generated story plan via Dynamic Contextual Engine.")

        story_dict["width"] = width
        story_dict["height"] = height
        story_dict["frame_length"] = frame_length
        story_dict["duration"] = actual_duration
        story_dict["fps"] = FPS

        summary_lines = [
            f"Title: {story_dict['story_title']}",
            f"Style: {story_dict['visual_style']} | Mood: {story_dict['pacing_mood']}",
            f"Resolution: {width}x{height} ({aspect_ratio})",
            f"Segments: {len(story_dict['segments'])} x {actual_duration}s ({frame_length} frames each)",
            f"Total Movie Duration: {len(story_dict['segments']) * actual_duration:.1f}s\n\nSegment Breakdown:",
            *(f"  [{s['segment_index']}] {s.get('summary', 'Segment ' + str(s['segment_index']))}" for s in story_dict["segments"])
        ]
        preview_lines = [f"=== SEGMENT {s['segment_index']} ===\n{s['prompt']}\n" for s in story_dict["segments"]]

        story_summary = "\n".join(summary_lines)
        character_bible = story_dict["character_bible"]
        prompts_preview = "\n".join(preview_lines)

        return (story_dict, story_summary, character_bible, prompts_preview, len(story_dict["segments"]), width, height, frame_length)
