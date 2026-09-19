# MiniMaxH3-StoryTelling

**Turn a single story idea or keyframe sequence into a full cinematic video with MiniMax FastH3 Lite & ComfyUI.**

Developed by **Geekatplay Studio - Vladimir Chopine** ([Website](https://www.geekatplay.com) | [GitHub](https://github.com/GeekatplayStudio/MiniMaxH3-StoryTelling))

---

## Overview

**Geekatplay StoryTeller** automates the entire journey from a text story premise to a multi-segment, high-fidelity video with native synchronized audio.

### Key Capabilities:
- **Local Ollama Story Director**: Translates your high-level story premise, character guidance, and style presets into a structured multi-segment script conforming strictly to the MiniMax H3 prompt guidelines (`base-en.txt` and `ref-en.txt`).
- **Seamless Offline Fallback**: If Ollama is offline or not installed, the built-in procedural continuity engine automatically takes over, ensuring the workflow never errors or fails.
- **Strict Character & World Continuity**: Automatically builds a persistent **Character Bible** (`<Subject 1>`, `<Subject 2>`) and stable speaker identifiers (`(S1)`, `(S2)`) with matching visual descriptions, attire, and vocal timbres across all segments.
- **FastVideo FastH3 8-Step Integration**: Uses the DMD2-distilled FastH3 8-step checkpoint (`fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`) for synchronized video + audio generation in just 8 sampling steps per segment.
- **Crash-Proof Batch Rendering & Resume**: Renders segments sequentially with automatic atomic file checkpointing. If interrupted or restarted, previously rendered segments are instantly reused.
- **Seamless Video & Audio Stitching**: Joins all segments into one continuous video, dropping boundary duplicate frames to keep seamless visual transitions and audio sync.
- **Optional Background Music Mixing**: Blends user-supplied music with ducking under dialogue and Foley sound effects.

---

## Nodes Included

1. **`Story Director (MiniMax / FastH3) - Geekatplay`** (`GAPStoryDirector`):
   - Formulates the narrative, character bible, aspect ratio canvas math (snapped to 32px), and per-segment multimodal prompts.
   - Outputs: `story_plan`, `story_summary`, `character_bible`, `prompts_preview`, `segment_count`, `width`, `height`, `frame_length`.
2. **`StoryTeller Batch Renderer (FastH3 8-Step) - Geekatplay`** (`GAPStoryTellerBatchRenderer`):
   - Executes the FastH3 8-step sampling across all segments, decodes video and audio VAEs, checkpoints segments, trims seams (`trim_seam_frame`), mixes optional music, and outputs the final combined `VIDEO`.
3. **`Story Segment Selector - Geekatplay`** (`GAPStorySegmentSelector`):
   - Extracts individual segments from a story plan for inspection or modular workflows.
4. **`Story Video Stitcher & Audio Mixer - Geekatplay`** (`GAPStoryVideoStitcher`):
   - Standalone video and audio stitching node with music ducking and crossfade controls.

### Keyframe Chain nodes

5. **`Keyframe Input - Image + Description`** (`GAPKeyframeInput`):
   - One image plus plain-language description of the shot that starts at it. Optional
     camera hint, spoken line, and per-keyframe duration override.
6. **`Segment Generator - Prompt & Length`** (`GAPSegmentGenerator`):
   - Rewrites the description into MiniMax H3 shot syntax and picks a **3s-10s** length
     from scene complexity. Outputs `segment`, `prompt_preview`, `duration`, `frame_length`.
7. **`Segment Manager - Keyframe Chain`** (`GAPSegmentManager`):
   - Grows its own segment inputs, chains them as first/last frame pairs, renders and
     stitches the whole thing. Outputs `video`, `frames`, `audio`, `segment_paths`,
     `chain_report`.

### Lipsync nodes

8. **`Lipsync Splitter - Word-Safe Segments`** (`GAPLipsyncSplitter`):
   - Cuts a video and its dialogue into MiniMax H3 sized segments without cutting through
     a spoken word. Outputs `plan`, `split_report`, `segment_count`, `segment_thumbnails`.
9. **`Lipsync Renderer - MiniMax H3`** (`GAPLipsyncRenderer`):
   - Re-renders each segment with the speech anchored as an audio guide and reassembles
     the full-length video. Outputs `video`, `frames`, `audio`, `segment_paths`,
     `render_report`, `segment_thumbnails`.

---

## Required Models

| Model Type | File Name | Target Directory |
|---|---|---|
| **Diffusion Model** | `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/` |
| **Text Encoder** | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `ComfyUI/models/text_encoders/` |
| **Video VAE** | `minimax_h3_video_vae_fp16.safetensors` | `ComfyUI/models/vae/` |
| **Audio VAE** | `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae/` |

---

## Requirements

- **ComfyUI** with native MiniMax H3 support (`comfy_extras/nodes_minimax_h3.py`) - the renderer
  reuses the core AV-latent and sigma-shift implementations.
- **torch** and **torchaudio** (both ship with ComfyUI).
- **Ollama** (optional). Without it the Story Director uses its built-in procedural fallback.
  Default model **`qwen3:8b`** - see *Choosing an Ollama model* below.
- **faster-whisper** (optional, lipsync only). Without it the Lipsync Splitter falls back to
  energy-based silence detection. `pip install faster-whisper`
- No pip packages beyond ComfyUI's own; the Ollama client uses the standard library.

The prompt reference guides live in `workflows/base-en.txt` and `workflows/ref-en.txt`. They ship
with a Geekatplay distillation of the MiniMax H3 rules; replace either file with the official
MiniMax text to have the Story Director feed that to Ollama instead. If a file is missing the
node still runs, only with less guidance.

---

## How Consistency Is Enforced

MiniMax H3 has no memory between segments. A bare `<Subject 1>` in segment 3 renders a
different person than segment 1, and the style drifts the same way. The Story Director
therefore rebuilds two locks and stamps them into **every** segment prompt:

- **Cast lock** - the full description of each subject, restated verbatim in every segment:
  `<Subject 1> is Dr. Maya Vance, late 20s botanist with braided auburn hair, beige utility
  jumpsuit, amber safety visor; voice (S1) is ...`
  Built from `character_hints`; write one subject per line as `Name: description`.
  Extra subjects stay silent unless the hint describes how they sound.
- **Style lock** - one sentence naming the lens, grade and lighting logic for the picked
  visual style and mood, closing every segment with the same negative clause.

Both locks are applied **after** the LLM writes, so they hold whether the shots came from
Ollama or from the offline fallback. The same pass also repairs what LLMs reliably drop:
a missing `[Shot 1]`, a `[Shot 2]` with no timestamp, `<d>` speech missing its `[English]`
tag, and second-person instruction leakage ("make sure to...").

## How Shot Variety Is Forced

A table of ten shot recipes (opening framing + camera move, cut framing + camera move) is
shuffled by the node `seed` and one recipe is assigned per segment, then handed to Ollama as
a binding shot plan. This is why segment 3 does not open on the same framing as segment 1.
The offline fallback walks the same table, plus shuffled action beats, dialogue lines, ambience
lines and music movements, so a 6-segment story never repeats a beat.

The `seed` is forwarded to Ollama (with `temperature 0.95`, `repeat_penalty 1.15`), so the
same seed reproduces the same storyboard and a new seed restages the whole story.

## Workflow 2: The Keyframe Chain

`workflows/keyframe_chain_fasth3.json` builds a continuous video out of images **you**
supply, using MiniMax H3's first-frame/last-frame mode.

```
LoadImage -> Keyframe Input -> Segment Generator ----.
LoadImage -> Keyframe Input -> Segment Generator ----+--> Segment Manager -> SaveVideo
LoadImage -> Keyframe Input -> Segment Generator ----'
```

### How the chain is paired

Each image is the **closing** frame of one segment and the **opening** frame of the next.
With four keyframes A, B, C, D the manager renders:

| segment | first frame | last frame |
|---|---|---|
| 1 | A | B |
| 2 | B | C |
| 3 | C | D |
| 4 | D | *none* (tail, image-to-video) |

The tail segment is what keeps the movie from ending on a freeze; turn `tail_segment` off
to stop on the last image instead. Because the shared keyframe is rendered at both ends of
every join, `trim_seam_frame` drops one copy - leave it on.

### Dynamic inputs

`segments` is an autogrow input. Connect a Segment Generator to the last empty slot and a
new empty slot appears; disconnect one and the slot goes away. Two minimum, twelve maximum.
**Slot order is story order.**

### Segment length, 3s to 10s

Each Segment Generator scores its own description - motion verbs, transformations, camera
moves, clause count, and any spoken line - and maps the score onto its `min_seconds` to
`max_seconds` range. A held portrait lands near 3s; "she runs across the rooftop, leaps the
gap, then rolls and comes up sprinting" lands near 9s. Every result is snapped to the
17k+5 frame grid at 24 fps and reported on `prompt_preview`.

Three overrides, highest priority first:

| Where | Control | Scope |
|---|---|---|
| Keyframe Input | `duration_override` (0 = off) | that segment |
| Segment Generator | `duration_mode: fixed` | that segment |
| Segment Manager | `duration_source` | all segments |

`duration_source: manager override` forces the manager's `segment_duration` everywhere;
`clamp generator to manager value` only shortens.

### What lives where

The Segment Generators write only **what moves** - action and camera. The Segment Manager
owns `visual_style`, `video_type`, `pacing_and_mood` and `aspect_ratio`, and stamps the same
style clause onto every segment prompt, exactly like the Story Director does. Change the
style once on the manager and the whole chain changes with it.

### Screen shape and screen size

These are two separate controls, on both the Story Director and the Segment Manager:

- **`aspect_ratio`** sets the SHAPE. On the manager, `match first keyframe` adopts the first
  image's proportions; any other choice forces that ratio and the images are fitted to it.
- **`frame_size`** sets the SIZE - how many pixels that shape is rendered at.

| frame_size | 16:9 | 9:16 | 21:9 | 1:1 |
|---|---|---|---|---|
| auto (H3 native) | 1344x768 | 768x1344 | 1344x672 | 1024x1024 |
| 1152 long edge | 1152x640 | 640x1152 | 1152x480 | 1024x1024 |
| 1024 long edge | 1024x576 | 576x1024 | 1024x448 | 1024x1024 |
| 896 long edge | 896x512 | 512x896 | 896x384 | 896x896 |
| 768 long edge | 768x448 | 448x768 | 768x320 | 768x768 |
| 640 long edge | 640x352 | 352x640 | 640x288 | 640x640 |

`custom (width x height below)` uses `custom_width` / `custom_height` directly.

Every result snaps both axes to 32 and clamps the total area to MiniMax H3's canvas limit
(768x1344 pixels), so no combination of ratio and size can produce a canvas the model cannot
render - a 1:1 frame asked for at 1344 comes back as 1024x1024 with a note on the console.
Smaller sizes render dramatically faster; use 640 or 768 to block out a story, then raise it.

## Lipsync with MiniMax H3

**Short answer: yes, the mechanism is there, and the Segment Manager exposes it.**

MiniMax H3 generates video and audio as one joint latent on a shared time axis. Core
ComfyUI's `MiniMaxH3AddGuide` can anchor an *audio* latent at a chosen frame
(`comfy_extras/nodes_minimax_h3.py`), and the DiT consumes it as `cond_audio_latents`
positioned on that same axis (`comfy/ldm/minimax/model.py`). So speech supplied as a guide
is time-aligned with the frames being denoised - which is exactly what lip motion needs. It
needs no extra model, no wav2lip, no post pass.

The manager exposes this as the autogrow input `lipsync_audio_0`, `lipsync_audio_1`... Each
track is anchored at frame 0 of the same-numbered segment, alongside that segment's image
keyframes.

Caveats, stated plainly:

- It is marked **experimental** because the mechanism is verified from the code, not from a
  measured render. How tightly the mouth tracks the audio is a property of the FastH3
  checkpoint, and an 8-step distilled model is the least likely variant to nail fine mouth
  detail. If sync matters, compare against base MiniMax H3 at 20+ steps.
- The guide is cropped to the segment's remaining duration, so a track longer than the
  segment is truncated rather than stretched.
- Anchored audio conditions the generated soundtrack too, so the native Foley is built
  around your track instead of competing with it.
- For dialogue you want *written* rather than supplied as a file, keep using the `dialogue`
  field on a Keyframe Input - that goes into the prompt as `<d>[English] ...</d>` and the
  model performs it.

## Every Segment Is Saved On Its Own

All three workflows write each finished segment to the ComfyUI output directory as a
clean numbered file, with a small `.json` beside it recording exactly how it was made:

```
output/lipsync/MiniMaxH3_segment_001.mp4
output/lipsync/MiniMaxH3_segment_001.json
output/lipsync/MiniMaxH3_segment_002.mp4
...
```

Those files are both the deliverable (use a segment on its own, re-cut them in an editor)
and the resume cache. On the next queue each segment's settings are compared against its
sidecar: unchanged means the file is loaded straight from disk, changed means it is
re-rendered and overwritten. Nothing accumulates and nothing goes stale.

### Re-rendering one bad segment

Every renderer has a **`rerender_segments`** field:

| you type | what happens |
|---|---|
| *(empty)* | everything finished is reused |
| `3` | segment 3 renders again, the rest is reused |
| `2, 5-7` | segments 2, 5, 6 and 7 render again |
| `all` | everything renders again |

So a single bad segment costs one segment's render time to fix, not the whole video.
Set `segment_prefix` to control where the files land.

## Workflow 3: Lipsync, Any Length

`workflows/lipsync_chain_fasth3.json` takes a video and a dialogue track of any length and
re-generates it with the mouth driven by the speech.

```
LoadVideo ---.
              +--> Lipsync Splitter --> Lipsync Renderer --> SaveVideo
LoadAudio ---'        (plan)              (per segment)
```

### The split is the hard part

Two constraints fight each other:

1. MiniMax H3 only renders frame counts on the **17k+5** grid - 5, 22, 39 ... 124, 141,
   158, 175, 192, 209 ... So segment lengths come from a discrete set, not a free choice.
2. A cut through the middle of a spoken word is audible and looks wrong.

So the splitter does not cut every N seconds. For each segment it evaluates **every**
grid-legal length inside `min_seconds`-`max_seconds`, scores where each candidate cut
would land, and takes the best one. Landing inside a word is penalised hard; landing in a
pause is rewarded in proportion to how much clear air surrounds it; drifting from
`target_seconds` costs a little. The result is segments of uneven length - 6.58s, 6.58s,
5.88s, 8.71s - each ending in a gap between words.

On a synthetic 47s track of 96 words, the word-aware split produced **zero** mid-word cuts
where a fixed 6s split produced five.

### Where the word timings come from

| split_method | source |
|---|---|
| `auto (whisper words, else silence)` | faster-whisper if installed, else the energy envelope |
| `whisper word timestamps` | faster-whisper, falling back if it is missing or fails |
| `silence detection` | the audio's short-time RMS - no ASR needed |
| `fixed length (may cut words)` | ignores content; for non-speech footage |

`faster-whisper` is optional. Without it the splitter uses the audio's own energy, which
finds pauses well enough for clean speech; install it (`pip install faster-whisper`) for
real word boundaries and for the transcript that goes into each segment's prompt. The
`base` model is plenty for boundaries.

### Timing is preserved exactly

- Segments are **disjoint** slices of one timeline, so unlike the keyframe chain nothing is
  trimmed at the joins - trimming would push every later word off its frame.
- The final segment is rounded up to the next grid length and trimmed back afterwards, so
  no padding reaches the output.
- `keep_source_audio` (on by default) puts your original dialogue on the finished video
  instead of the model's generated audio, which guarantees the words stay where they were.
- A source that is not 24 fps is resampled to 24 fps first, so a frame index means the same
  instant in the source and in the render.

Verified on a 47s source: 7 segments in, 1128 frames and 47.00s back out, matching the
input frame for frame.

### Reviewing the result

`split_report` lists every segment with its time range, the words in it, and whether the
cut was **clean**, **ok** or **IN WORD**. `segment_thumbnails` on both nodes feeds a
Preview Image node showing the first frame of every segment, so a bad one is easy to spot -
then put its number in `rerender_segments` and queue again.

### Cost

Every segment is a full FastH3 render. A 3-minute video at ~6.5s per segment is roughly
28 renders. Start with a short clip.

## Choosing an Ollama Model

The default is **`qwen3:8b`**, chosen by measurement rather than reputation. Each candidate
wrote the same 4-segment storyboard from the same premise and seed, scored on what this
workflow actually needs - valid JSON, `[Shot 1]`, a timed `[Shot 2]`, `<Subject N>` tags,
`<d>[English]>` dialogue, camera grammar (motion + amplitude + speed), shot-size variety,
and no instruction leakage:

| model | score | time | VRAM | notes |
|---|---|---|---|---|
| `qwen3:30b` | **95/100** | 231s | 18.6 GB | only model to write dialogue unprompted (4/4); subject tags 2/4 |
| `qwen3.8:latest` | 90/100 | 246s | 17.7 GB | tags 4/4, very verbose (211 words/segment) |
| **`qwen3:8b`** | **90/100** | **16s** | **5.2 GB** | tags 4/4, cuts 4/4, tight 62 words/segment |
| `gemma3:12b` | 72/100 | 36s | 8.1 GB | wrote **no** timed cuts at all |
| `gpt-oss:20b` | - | 49s | 13.8 GB | HTTP 500 from Ollama; fell back to the procedural writer |

`qwen3:8b` ties the newest 27B model on quality while being **14x faster** and using **a third
of the VRAM**. That last column is the one that matters on a single card: the Story Director
runs immediately before a MiniMax H3 render, and 5GB leaves room for it where 18GB does not.

`qwen3:30b` scored 5 points higher, entirely on dialogue. If you want spoken lines and do not
mind ~4 minutes of writing per storyboard, pick it from the dropdown - the workflow needs no
other change. Treat the 90-vs-95 gap as indicative, not precise: it is one sample per model.

**Known gap:** none of the 8B-27B models volunteer dialogue unless the premise asks for it.
Tightening the instruction to demand dialogue was tried and measured - it lifted dialogue from
0 to 1 segment but collapsed timed cuts from 4 to 0 and dropped the score to 77, so it was
reverted. Put the spoken line in your premise or character hints instead, or use `qwen3:30b`.

`ollama_model` is a **dropdown, not a text field**. It is built when ComfyUI loads the node
list, from a live query to your Ollama:

- models **you already have** are listed first, in the order Ollama reports them
- then the recommended models you could pull: `qwen3:8b`, `qwen3:14b`, `qwen3:30b`,
  `qwen3:4b`, `qwen3:30b-a3b`, `llama3.1:8b`, `gemma3:12b`, `mistral-nemo:12b`, `qwen2.5:7b`
- then `custom (use ollama_model_custom)`

The tooltip on the widget names exactly which of them are installed and which are not.
Refresh the browser after pulling a model to see it appear. Embedding and base models
(`nomic-embed-text`, `*-base`) are filtered out of the list - they cannot write a
storyboard.

### If the model isn't downloaded

`auto_pull_model` is on by default. The first time you use a model that Ollama does not
have, the node pulls it through Ollama's own API and then continues. It happens once and
prints progress to the console.

Turn `auto_pull_model` **off** and nothing is ever downloaded; instead the node falls back
to the best model you already have, working down the recommended list, and tells you what
it used and how to get the one you asked for:

```
[Geekatplay Studio] 'qwen3:8b' is not installed; falling back to 'qwen3:30b'.
Turn on auto_pull_model to download 'qwen3:8b' instead, or run: ollama pull qwen3:8b
```

If Ollama is not running at all, or has no usable model, the node says so once and uses its
built-in procedural writer. **The queue never fails over a missing model.**

### Anything not in the list

Type it into `ollama_model_custom`. A non-empty value always wins over the dropdown, so a
local fine-tune or an unusual tag works without touching the code.

### One model load per queue, and none during the render

The prompt-writing nodes and the renderers have opposite needs, so they are handled
separately:

- **Story Director / Segment Generator** - `keep_model_loaded` (default on) keeps the LLM
  resident between calls. The keyframe chain has one generator per keyframe, so this is the
  difference between loading the model once and loading it for every segment.
- **Batch Renderer / Segment Manager / Lipsync Renderer** - `free_ollama_vram` (default on)
  asks Ollama to drop everything it is holding **once**, immediately before the diffusion
  models load, and reports what it freed:

  ```
  [Geekatplay Studio] Freed 17.7 GB of VRAM from Ollama before rendering (qwen3:30b).
  ```

This matters because Ollama's own default is to hold a model in VRAM for five minutes after
answering. On a single 24GB card a resident 18GB LLM and a FastH3 render do not both fit.

Both workflows already run in the right order - every prompt is written before any video is
rendered. The Story Director writes all segments in a single call; in the keyframe chain the
Segment Generators all complete before the Segment Manager starts, because it depends on
their outputs.

### Note on Qwen3 and thinking

Qwen3 is a reasoning model: left alone it spends its output budget on a `<think>` block
before answering, which wastes tokens and buries the JSON. The node sends `think: false`
with every request, and retries without that field if your Ollama build is too old to
accept it. Any `<think>` block that still comes back is stripped before the JSON is parsed,
including an unclosed one, so a thinking model cannot break the parse.

### Saved workflows stay valid

Dropdown entries are plain model names with no status text attached, so a workflow saved on
a machine that has `qwen3:8b` still opens cleanly on one that does not - it just pulls or
falls back at render time.

## Sampler Settings That Matter

FastH3 needs the MiniMax H3 flow shifts, exactly like the reference
`video_fastvideo_fasth3_t2v.json` workflow. The batch renderer applies them internally through
`shift_video` (default **10.0**) and `shift_audio` (default **3.0**); leaving them at the plain
model defaults produces washed-out, unconverged 8-step output.

| Setting | Value |
|---|---|
| `steps` | 8 |
| `sampler_name` | `euler` |
| `scheduler` | `simple` |
| `shift_video` | 10.0 |
| `shift_audio` | 3.0 |

## Music Mix Modes

`blend with video audio`, `ducking under video audio`, `replace video audio` - the same three
names are used by both `GAPStoryTellerBatchRenderer` and `GAPStoryVideoStitcher`.

---

## Ready-to-Use Workflow

| Workflow | What it does |
|---|---|
| `workflows/storyteller_fasth3_full.json` | One story premise to a full multi-segment movie. |
| `workflows/keyframe_chain_fasth3.json` | Your own images, chained as first/last frame pairs. |
| `workflows/lipsync_chain_fasth3.json` | A video plus dialogue, re-rendered in sync at any length. |

Open either directly in ComfyUI.

`workflows/video_fastvideo_fasth3_t2v.json` and `workflows/video_fastvideo_fasth3_i2v.json`
are the ComfyUI reference FastH3 graphs, kept here so the StoryTeller settings can be
compared against them.
