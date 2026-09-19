"""
Geekatplay Studio - Vladimir Chopine
Keyframe Input for the StoryTeller Segment Chain
Website: https://www.geekatplay.com

One image plus what should happen at it. These feed the Segment Generator, which
turns the plain-language description into MiniMax H3 prompt syntax, and then the
Segment Manager, which chains the keyframes into first/last-frame video segments.
"""


class GAPKeyframeInput:
    """Geekatplay Studio - Vladimir Chopine.
    Pairs one keyframe image with a plain-language description of the shot it
    starts. Connect a LoadImage here and write what you want to happen."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The keyframe. Feed a LoadImage node."
                }),
                "description": ("STRING", {
                    "multiline": True,
                    "default": "The camera holds on the subject as they turn toward the light.",
                    "tooltip": "Plain language. What happens in the shot that starts at this image? "
                               "The Segment Generator rewrites this into MiniMax H3 prompt syntax."
                }),
            },
            "optional": {
                "keyframe_label": ("STRING", {
                    "default": "",
                    "tooltip": "Optional name for this keyframe, shown in the manager's report."
                }),
                "camera_hint": ("STRING", {
                    "default": "",
                    "tooltip": "Optional camera instruction for the shot that starts here, e.g. "
                               "'slow push in' or 'handheld follow'. Left empty, the generator picks one."
                }),
                "duration_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Seconds for the segment that starts at this keyframe. "
                               "0 means let the Segment Generator decide from scene complexity."
                }),
                "dialogue": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional spoken line for this shot. Written into the prompt as "
                               "<d>[English] ...</d> so MiniMax H3 generates it as native speech."
                }),
            }
        }

    RETURN_TYPES = ("GAP_KEYFRAME",)
    RETURN_NAMES = ("keyframe",)
    FUNCTION = "build_keyframe"
    CATEGORY = "Geekatplay/StoryTeller"
    DESCRIPTION = "One keyframe image plus a description of the shot that starts at it."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def build_keyframe(self, image, description, keyframe_label="", camera_hint="",
                       duration_override=0.0, dialogue="", **kwargs):
        if image is None or image.shape[0] == 0:
            raise ValueError("GAPKeyframeInput needs an image.")

        keyframe = {
            "image": image[:1],
            "description": (description or "").strip(),
            "label": (keyframe_label or "").strip(),
            "camera_hint": (camera_hint or "").strip(),
            "duration_override": float(duration_override or 0.0),
            "dialogue": (dialogue or "").strip(),
            "height": int(image.shape[1]),
            "width": int(image.shape[2]),
        }
        return (keyframe,)
