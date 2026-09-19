"""
Geekatplay Studio - Vladimir Chopine
StorySegmentSelector for MiniMax FastH3 Lite & ComfyUI
Website: https://www.geekatplay.com

Extracts individual story segments, prompts, durations, soundscapes, and
music notes from a GAP_STORY_PLAN for modular workflows.
"""


class GAPStorySegmentSelector:
    """Geekatplay Studio - Vladimir Chopine.
    Extracts a specific shot/segment from a GAP_STORY_PLAN for modular workflows.
    Allows routing each segment directly into MiniMaxH3ImageToVideo or other ComfyUI nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_plan": ("GAP_STORY_PLAN",),
                "segment_index": ("INT", {
                    "default": 1, "min": 1, "max": 20, "step": 1,
                    "tooltip": "1-based index of the segment to extract."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT", "INT", "STRING", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("prompt", "duration", "frame_length", "width", "height", "soundscape", "music_notes", "is_last_segment", "summary")
    FUNCTION = "select_segment"
    CATEGORY = "Geekatplay/StoryTeller"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def select_segment(self, story_plan, segment_index=1, **kwargs):
        if not story_plan or "segments" not in story_plan or not story_plan["segments"]:
            raise ValueError("Invalid story plan supplied to GAPStorySegmentSelector.")

        segments = story_plan["segments"]
        total = len(segments)
        idx = max(1, min(segment_index, total)) - 1
        seg = segments[idx]

        width = story_plan.get("width", 1344)
        height = story_plan.get("height", 768)
        frame_length = seg.get("frame_count", story_plan.get("frame_length", 124))
        duration = float(seg.get("duration", story_plan.get("duration", 5.0)))
        prompt = seg.get("prompt", "")
        soundscape = seg.get("soundscape", "")
        music_notes = seg.get("music", "")
        is_last = (idx == total - 1)
        summary = f"Segment {idx + 1}/{total}: {seg.get('summary', '')}"

        return (prompt, duration, frame_length, width, height, soundscape, music_notes, is_last, summary)
