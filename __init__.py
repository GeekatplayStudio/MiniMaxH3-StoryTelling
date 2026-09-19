"""
ComfyUI-Geekatplay-StoryTeller
Geekatplay Studio - Vladimir Chopine
Website: https://www.geekatplay.com

Cinematic story-to-video generation suite for MiniMax FastH3 Lite & ComfyUI.
Turns a single narrative premise into a multi-segment continuous cinematic video
with character consistency, rich camera motion, and synchronized native audio -
or chains your own keyframe images into one continuous first/last-frame movie.
"""

from .story_director import GAPStoryDirector
from .story_segment_selector import GAPStorySegmentSelector
from .story_batch_renderer import GAPStoryTellerBatchRenderer
from .story_stitcher import GAPStoryVideoStitcher
from .story_keyframe_input import GAPKeyframeInput
from .story_segment_generator import GAPSegmentGenerator
from .story_segment_manager import GAPSegmentManager
from .story_lipsync import GAPLipsyncSplitter, GAPLipsyncRenderer

NODE_CLASS_MAPPINGS = {
    "GAPStoryDirector": GAPStoryDirector,
    "GAPStoryTellerBatchRenderer": GAPStoryTellerBatchRenderer,
    "GAPStorySegmentSelector": GAPStorySegmentSelector,
    "GAPStoryVideoStitcher": GAPStoryVideoStitcher,
    "GAPKeyframeInput": GAPKeyframeInput,
    "GAPSegmentGenerator": GAPSegmentGenerator,
    "GAPSegmentManager": GAPSegmentManager,
    "GAPLipsyncSplitter": GAPLipsyncSplitter,
    "GAPLipsyncRenderer": GAPLipsyncRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GAPStoryDirector": "Story Director - MiniMax H3 (Geekatplay Studio)",
    "GAPStoryTellerBatchRenderer": "Story Batch Renderer - FastH3 (Geekatplay Studio)",
    "GAPStorySegmentSelector": "Story Segment Selector (Geekatplay Studio)",
    "GAPStoryVideoStitcher": "Story Video Stitcher (Geekatplay Studio)",
    "GAPKeyframeInput": "Keyframe Input - Image + Description (Geekatplay Studio)",
    "GAPSegmentGenerator": "Segment Generator - Prompt & Length (Geekatplay Studio)",
    "GAPSegmentManager": "Segment Manager - Keyframe Chain (Geekatplay Studio)",
    "GAPLipsyncSplitter": "Lipsync Splitter - Word-Safe Segments (Geekatplay Studio)",
    "GAPLipsyncRenderer": "Lipsync Renderer - MiniMax H3 (Geekatplay Studio)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "GAPStoryDirector",
    "GAPStoryTellerBatchRenderer",
    "GAPStorySegmentSelector",
    "GAPStoryVideoStitcher",
    "GAPKeyframeInput",
    "GAPSegmentGenerator",
    "GAPSegmentManager",
    "GAPLipsyncSplitter",
    "GAPLipsyncRenderer",
]
