"""The sitting dog at the center of the crosswalk.

The dog stays put, loops its idle cycle and turns to watch passing people:
it picks the facing-away cycle (dog_forward) or the facing-camera cycle
(dog_backward), mirrors it toward the side it is looking at, and adds a slight
tilt. People treat it as an obstacle (see crowd.Obstacle).
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from PIL import Image

import crowd

# facing (relative to the camera) -> (folder, file prefix)
SETS = {
    "away": ("dog_forward", "dog_forward"),
    "toward": ("dog_backward", "dog_backwards"),
}
WORK_HEIGHT = 360  # pixel height of the dog (top of head to feet) in cached sprites
PAD = 0.3  # transparent margin (fraction of height) so tilting never clips
GAZE_WINDOW = 8  # past frames blended into the gaze to keep it smooth
GAZE_DECAY = 0.75
IDLE_GAZE = 0.15  # pull toward the camera when nobody is near

_ASSETS = None


def load_assets():
    """Crop, scale and mirror both dog cycles (cached).

    Returns {(facing, mirrored): (frames, (anchor_x, anchor_y))}. Unmirrored
    sprites look toward screen-left.
    """
    global _ASSETS
    if _ASSETS is not None:
        return _ASSETS
    raw, boxes = {}, {}
    for facing, (folder, prefix) in SETS.items():
        paths = sorted((crowd.ANIM_DIR / folder).glob(f"{prefix}*.png"))
        if not paths:
            raise FileNotFoundError(f"No dog frames found in {crowd.ANIM_DIR / folder}")
        raw[facing] = [Image.open(path).convert("RGBA") for path in paths]
        bbox = None
        for im in raw[facing]:
            bbox = crowd._union(bbox, im.getbbox())
        boxes[facing] = bbox
    # One shared scale keeps both cycles the same size on screen.
    ref_h = max(b[3] - b[1] for b in boxes.values())
    scale = WORK_HEIGHT / ref_h
    pad = round(ref_h * PAD)
    assets = {}
    for facing, frames in raw.items():
        left, top, right, bottom = boxes[facing]
        box = (left - pad, top - pad, right + pad, bottom + pad)
        size = (round((box[2] - box[0]) * scale), round((box[3] - box[1]) * scale))
        sprites = [im.crop(box).resize(size, Image.LANCZOS) for im in frames]
        # Anchor at the center of the paws: the lowest opaque rows of the cycle.
        alpha = np.max([np.asarray(s.getchannel("A")) for s in sprites], axis=0)
        rows = np.nonzero(alpha.max(1) > 64)[0]
        foot_y = int(rows[-1])
        band = alpha[max(0, foot_y - round(WORK_HEIGHT * 0.06)):foot_y + 1]
        cols = np.nonzero(band.max(0) > 64)[0]
        foot_x = (cols[0] + cols[-1]) / 2
        assets[facing, False] = sprites, (foot_x, foot_y)
        assets[facing, True] = ([s.transpose(Image.FLIP_LEFT_RIGHT) for s in sprites],
                                (size[0] - foot_x, foot_y))
    _ASSETS = assets
    return assets


@dataclass
class Dog:
    WORK_HEIGHT: ClassVar[int] = WORK_HEIGHT
    height: float  # world-unit height (top of head to paws)
    clearance: float  # radius people keep from the dog
    max_tilt: float  # degrees of lean when looking fully sideways
    hold: int  # output frames each drawing is held for
    x: float = 0.0
    y: float = 0.0

    def obstacle(self):
        return crowd.Obstacle(self.x, self.y, self.clearance)

    def gaze(self, cam, people, t):
        """Unit (right, away) camera-relative direction the dog looks in at frame t."""
        obstacle = self.obstacle()
        sigma = max(2.0, 3 * self.clearance)
        gx = gy = 0.0
        total = 0.0
        for k in range(GAZE_WINDOW):
            decay = GAZE_DECAY ** k
            total += decay
            for person in people:
                x, y = person.ground_pose(t - k, obstacle)
                dx, dy = x - self.x, y - self.y
                right = dx * cam.ca - dy * cam.sa
                away = dx * cam.sa + dy * cam.ca
                d = math.hypot(right, away)
                if d < 1e-6:
                    continue
                w = decay * math.exp(-(d / sigma) ** 2)
                gx += w * right / d
                gy += w * away / d
        gy -= IDLE_GAZE * total
        n = math.hypot(gx, gy)
        return gx / n, gy / n

    def sprite(self, cam, people, t):
        """(RGBA sprite, foot anchor) for frame t."""
        gx, gy = self.gaze(cam, people, t)
        frames, anchor = load_assets()["away" if gy > 0 else "toward", gx > 0]
        frame = frames[(t // max(1, self.hold)) % len(frames)]
        tilt = self.max_tilt * gx
        if abs(tilt) > 0.05:
            # PIL rotates counter-clockwise; looking right leans clockwise.
            frame = frame.rotate(-tilt, resample=Image.BICUBIC, center=anchor)
        return frame, anchor
