"""The sitting dog at the center of the crosswalk.

Normally the dog stays put, loops its idle cycle and watches passing people:
it picks the facing-away cycle (dog_forward) or the facing-camera cycle
(dog_backward), mirrors it toward the side it is looking at, and adds a slight
tilt. An optional custom-pedestrian interaction adds excited marks while petting,
then a walking cycle as the dog follows the departing pedestrian.
People treat it as an obstacle (see crowd.Obstacle).
"""

import math
import random
from dataclasses import dataclass
from functools import lru_cache
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
_BLINK_GEOMETRY = None


def load_assets():
    """Crop, scale and mirror both dog cycles (cached).

    Returns {(facing, mirrored): (frames, (anchor_x, anchor_y))}. Unmirrored
    sprites look toward screen-left.
    """
    global _ASSETS, _BLINK_GEOMETRY
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
        if facing == "toward":
            _BLINK_GEOMETRY = box, size, [im.size for im in frames]
    _ASSETS = assets
    return assets


@lru_cache(maxsize=1)
def load_blink_assets():
    assets = load_assets()
    box, size, original_sizes = _BLINK_GEOMETRY
    folder = crowd.ANIM_DIR / "dog_backwards_blink"
    paths = sorted(folder.glob("dog_backwards_blink*.png"))
    if len(paths) != len(original_sizes):
        raise ValueError(f"Expected {len(original_sizes)} dog blink frames in {folder}, "
                         f"found {len(paths)}")
    sprites = []
    for path, original_size in zip(paths, original_sizes):
        with Image.open(path) as image:
            if image.size != original_size:
                raise ValueError(f"Dog blink frame {path} does not match the normal frame size")
            sprites.append(image.convert("RGBA").crop(box).resize(size, Image.LANCZOS))
    return {
        False: (sprites, assets["toward", False][1]),
        True: ([sprite.transpose(Image.FLIP_LEFT_RIGHT) for sprite in sprites],
               assets["toward", True][1]),
    }


def is_blinking(t, fps):
    """Return whether frame t falls in a deterministic, occasional blink."""
    fps = max(1, fps)
    if t < 0:
        return False
    bucket = t // (3 * fps)
    duration = max(1, round(0.15 * fps))
    for index in (bucket - 1, bucket):
        if index >= 0:
            start = (3 * index + 2) * fps + random.Random(f"dog-blink:{index}").randrange(fps)
            if start <= t < start + duration:
                return True
    return False


@lru_cache(maxsize=1)
def load_interaction_assets():
    """Keep overlays registered on the source canvas, with idle body scale."""
    load_assets()
    _, base_size, _ = _BLINK_GEOMETRY
    base_box = _BLINK_GEOMETRY[0]
    scale = base_size[1] / (base_box[3] - base_box[1])
    results = {}
    for kind, folder, prefix in (
            ("excited", "dog_backwards_blink", "dog_excited"),
            ("walk", "dog_walk", "dog_walk")):
        paths = sorted((crowd.ANIM_DIR / folder).glob(f"{prefix}*.png"))
        if not paths:
            raise ValueError(f"No {kind} dog frames in {crowd.ANIM_DIR / folder}")
        frames = []
        for path in paths:
            with Image.open(path) as image:
                if image.size != (1000, 1000):
                    raise ValueError(f"Dog interaction frame must use the 1000x1000 canvas: {path}")
                frames.append(image.convert("RGBA"))
        if kind == "excited":
            base_paths = sorted((crowd.ANIM_DIR / "dog_backward").glob("dog_backwards*.png"))
            bases = []
            for path in base_paths:
                with Image.open(path) as image:
                    bases.append(image.convert("RGBA"))
            # Use the idle paws, not the excited marks, as the ground anchor.
            ax, ay = load_assets()["toward", False][1]
            source_anchor = (base_box[0] + ax / scale, base_box[1] + ay / scale)
            frames = [Image.alpha_composite(bases[i % len(bases)], frames[i % len(frames)])
                      for i in range(math.lcm(len(bases), len(frames)))]
        else:
            alpha = np.max([np.asarray(im.getchannel("A")) for im in frames], axis=0)
            rows = np.nonzero(alpha.max(1) > 64)[0]
            if not rows.size:
                raise ValueError("The dog walking animation is fully transparent.")
            fy = int(rows[-1])
            cols = np.nonzero(alpha[max(0, fy - 20):fy + 1].max(0) > 64)[0]
            source_anchor = ((cols[0] + cols[-1]) / 2, fy)
        box = None
        for frame in frames:
            box = crowd._union(box, frame.getbbox())
        box = (box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4)
        size = (round((box[2] - box[0]) * scale), round((box[3] - box[1]) * scale))
        sprites = [im.crop(box).resize(size, Image.LANCZOS) for im in frames]
        anchor = ((source_anchor[0] - box[0]) * scale,
                  (source_anchor[1] - box[1]) * scale)
        results[kind, False] = sprites, anchor
        results[kind, True] = ([s.transpose(Image.FLIP_LEFT_RIGHT) for s in sprites],
                               (size[0] - anchor[0], anchor[1]))
    return results


@dataclass
class Dog:
    WORK_HEIGHT: ClassVar[int] = WORK_HEIGHT
    height: float  # world-unit height (top of head to paws)
    clearance: float  # radius people keep from the dog
    max_tilt: float  # degrees of lean when looking fully sideways
    hold: int  # output frames each drawing is held for
    x: float = 0.0
    y: float = 0.0
    blink_enabled: bool = True
    fps: int = 12

    def obstacle(self):
        return crowd.Obstacle(self.x, self.y, self.clearance)

    def gaze(self, cam, people, t, obstacle=None):
        """Unit (right, away) camera-relative direction the dog looks in at frame t."""
        obstacle = self.obstacle() if obstacle is None else obstacle
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

    def sprite(self, cam, people, t, *, obstacle=None):
        """(RGBA sprite, foot anchor) for frame t."""
        gx, gy = self.gaze(cam, people, t, obstacle)
        facing = "away" if gy > 0 else "toward"
        mirrored = gx > 0
        if facing == "toward" and self.blink_enabled and is_blinking(t, self.fps):
            frames, anchor = load_blink_assets()[mirrored]
        else:
            frames, anchor = load_assets()[facing, mirrored]
        frame = frames[(t // max(1, self.hold)) % len(frames)]
        tilt = self.max_tilt * gx
        if abs(tilt) > 0.05:
            # PIL rotates counter-clockwise; looking right leans clockwise.
            frame = frame.rotate(-tilt, resample=Image.BICUBIC, center=anchor)
        return frame, anchor

    def interaction_sprite(self, cam, people, t, interaction, obstacle=None):
        if t < interaction.pet_start:
            return self.sprite(cam, people, t, obstacle=obstacle)
        if interaction.petting(t):
            kind, mirrored = "excited", False
            elapsed = t - interaction.pet_start
        else:
            _, _, dx, dy, distance = interaction.dog_pose(t)
            if distance <= 0:
                return self.sprite(cam, people, t, obstacle=obstacle)
            kind, mirrored = "walk", dx * cam.ca - dy * cam.sa > 0
            first_step = math.floor(interaction.dog_delay * interaction.fps / interaction.speed) + 1
            elapsed = (max(0, t - interaction.return_start - first_step)
                       if 0 < distance < interaction.dog_distances[-1] else 0)
        frames, anchor = load_interaction_assets()[kind, mirrored]
        return frames[(elapsed // max(1, self.hold)) % len(frames)], anchor
