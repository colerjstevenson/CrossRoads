"""Randomized pedestrian crowds for the crosswalk generator.

Loads the layered 12-frame walk cycles from Animations/ (legs, arms, head,
hair_male / hair_female), recolors each layer per person while preserving the
crayon texture, and renders the crowd with perspective scaling through the
generator's camera.
"""

import colorsys
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ANIM_DIR = Path(__file__).resolve().parent / "Animations"
FRAME_COUNT = 12
WORK_HEIGHT = 480  # pixel height the cached sprites are pre-scaled to
STRIDE_FACTOR = 0.85  # world distance walked per cycle, in person heights

# direction -> (folder, file prefix), relative to the camera.
DIRECTIONS = {
    "away": ("walk_forward", "walk_forward"),
    "toward": ("walk_backwards", "walk_backward"),
}
HAIR_LAYERS = ("hair_male", "hair_female")
BODY_LAYERS = ("legs", "arms", "head")

SKIN_TONES = [
    (255, 224, 196), (240, 200, 170), (224, 177, 144), (198, 134, 94),
    (161, 102, 67), (124, 74, 46), (90, 56, 37),
]
HAIR_COLORS = [
    (28, 22, 20), (60, 40, 25), (100, 65, 35), (150, 100, 50),
    (205, 165, 95), (145, 70, 35), (130, 130, 130), (225, 218, 200),
]

_ASSETS = None


# ------------------------------------------------------------- asset loading


def load_assets():
    """Load, crop and normalize all walk-cycle layers (cached)."""
    global _ASSETS
    if _ASSETS is not None:
        return _ASSETS
    assets = {}
    for direction, (folder, prefix) in DIRECTIONS.items():
        raw = {}
        bbox = None
        legs_bbox = None
        for layer in BODY_LAYERS + HAIR_LAYERS:
            frames = []
            for i in range(1, FRAME_COUNT + 1):
                path = ANIM_DIR / folder / layer / f"{prefix}_{layer}{i:04d}.png"
                im = Image.open(path).convert("RGBA")
                frames.append(im)
                bb = im.getbbox()
                if bb:
                    bbox = _union(bbox, bb)
                    if layer == "legs":
                        legs_bbox = _union(legs_bbox, bb)
            raw[layer] = frames
        left, top, right, bottom = bbox
        scale = WORK_HEIGHT / (bottom - top)
        width = max(1, round((right - left) * scale))
        layers = {}
        for layer, frames in raw.items():
            arrs = []
            vals = []
            for im in frames:
                im = im.crop(bbox).resize((width, WORK_HEIGHT), Image.LANCZOS)
                a = np.asarray(im, dtype=np.float32)
                arrs.append(a)
                mask = a[..., 3] > 64
                if mask.any():
                    vals.append(a[..., :3].max(-1)[mask])
            # Brightness of a typical fully-colored stroke; recoloring maps it
            # to the target color so the crayon texture is preserved.
            v_ref = max(float(np.percentile(np.concatenate(vals), 95)), 1.0)
            frames_norm = []
            for a in arrs:
                norm = (a[..., :3].max(-1) / v_ref).astype(np.float16)
                frames_norm.append((norm, a[..., 3].astype(np.uint8)))
            layers[layer] = frames_norm
        anchor = (
            ((legs_bbox[0] + legs_bbox[2]) / 2 - left) * scale,  # foot center x
            (legs_bbox[3] - top) * scale,  # ground line y
        )
        assets[direction] = {"layers": layers, "size": (width, WORK_HEIGHT), "anchor": anchor}
    _ASSETS = assets
    return assets


def _union(a, b):
    if a is None:
        return b
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


# ------------------------------------------------------------ randomization


def _jitter(rng, color, amount):
    return tuple(int(min(255, max(0, c + rng.uniform(-amount, amount)))) for c in color)


def _clothing_color(rng):
    hue = rng.random()
    sat = rng.uniform(0.25, 0.95)
    val = rng.uniform(0.35, 0.95)
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, sat, val)), hue


def random_look(rng):
    """Random layer colors: shirt, pants, skin tone, hair."""
    shirt, shirt_hue = _clothing_color(rng)
    while True:
        pants, pants_hue = _clothing_color(rng)
        d = abs(shirt_hue - pants_hue)
        if min(d, 1 - d) > 0.08:
            break
    return {
        "legs": pants,
        "arms": shirt,
        "head": _jitter(rng, rng.choice(SKIN_TONES), 8),
        "hair": _jitter(rng, rng.choice(HAIR_COLORS), 10),
    }


def _build_frames(direction, colors, female, mirror):
    """Recolor every layer and composite the 12 person frames."""
    data = load_assets()[direction]
    width, height = data["size"]
    hair = "hair_female" if female else "hair_male"
    order = [("legs", colors["legs"]), ("arms", colors["arms"]),
             ("head", colors["head"]), (hair, colors["hair"])]
    frames = []
    for i in range(FRAME_COUNT):
        comp = Image.new("RGBA", (width, height))
        for layer, color in order:
            norm, alpha = data["layers"][layer][i]
            rgb = np.clip(norm[..., None].astype(np.float32) * np.asarray(color, dtype=np.float32), 0, 255).astype(np.uint8)
            comp = Image.alpha_composite(comp, Image.fromarray(np.dstack([rgb, alpha])))
        if mirror:
            comp = comp.transpose(Image.FLIP_LEFT_RIGHT)
        frames.append(comp)
    ax, ay = data["anchor"]
    if mirror:
        ax = width - ax
    return frames, (ax, ay)


@dataclass(frozen=True)
class WalkRoute:
    along_x: bool
    start: float
    end: float
    half_depth: float


@dataclass(frozen=True)
class Obstacle:
    """A ground-plane point people steer around."""
    x: float
    y: float
    radius: float

    def lateral_offset(self, route: WalkRoute, along: float, lane: float) -> float:
        """Sideways shift that carries a walker at (along, lane) around the obstacle."""
        center_along, center_lane = (self.x, self.y) if route.along_x else (self.y, self.x)
        r = self.radius
        if r <= 0:
            return 0.0
        gap = lane - center_lane
        # Lanes within `spread` are squeezed out to [r, spread], keeping walkers ordered.
        spread = 1.6 * r
        if abs(gap) >= spread:
            return 0.0
        side = 1.0 if gap >= 0 else -1.0
        need = r + abs(gap) * (spread - r) / spread - abs(gap)
        ramp = 2.0 * r
        d = abs(along - center_along)
        if d >= r + ramp:
            return 0.0
        u = 1.0 if d <= r else (r + ramp - d) / ramp
        return side * need * u * u * (3 - 2 * u)


@dataclass
class Person:
    route: WalkRoute
    lane: float
    offset: float
    dir_sign: int
    rate: float  # walk-cycle frames advanced per output frame
    phase: float  # walk-cycle start offset
    height: float  # world-unit height
    frames: list  # 12 composited RGBA sprites
    anchor: tuple  # (x, y) foot anchor in sprite pixels
    entry: float  # signed camera-horizontal extension at route.start
    exit: float  # signed camera-horizontal extension at route.end
    right: tuple[float, float]  # camera-right direction on the ground

    def ground_pose(self, t, obstacle: Obstacle | None = None):
        """Ground anchor, wrapping only after a frame fully outside each edge."""
        route = self.route
        length = abs(self.entry) + route.end - route.start + abs(self.exit)
        step = self.rate / FRAME_COUNT * STRIDE_FACTOR * self.height
        # A step's worth of hidden hold at each endpoint prevents a fast
        # walker from jumping between two partially visible positions.
        distance = min(length, max(0.0, (self.offset + self.dir_sign * step * t)
                                   % (length + 2 * step) - step))
        along = route.start + distance - abs(self.entry)
        side = 0.0
        if along < route.start:
            side = math.copysign(route.start - along, self.entry)
            along = route.start
        elif along > route.end:
            side = math.copysign(along - route.end, self.exit)
            along = route.end
        x, y = _lane_point(route, along, self.lane, obstacle)
        return x + side * self.right[0], y + side * self.right[1]


def _lane_point(route, along, lane, obstacle):
    if obstacle is not None:
        lane += obstacle.lateral_offset(route, along, lane)
    return (along, lane) if route.along_x else (lane, along)


def _walker_projection(cam, x, y, height):
    cx, cy, depth = cam.to_cam(x, y)
    # Keep the drawn figure continuous instead of dropping it when a close
    # head hits the near plane or a top-down view foreshortens it to nothing.
    z = max(depth, cam.dist / 4)
    fx, fy = cam.project((cx, cy, z))
    _, head_y = cam.project((cx, cy + height * cam.up[2],
                             max(z / 2, z + height * cam.fwd[2])))
    min_height = 0.5 * cam.focal * height / z
    pixel_h = min(3 * min(cam.w, cam.h), max(min_height, fy - head_y))
    return depth, fx, fy, pixel_h


def _sprite_box(sprite, anchor, fx, fy, scale):
    ax, ay = anchor
    return (round(fx - ax * scale), round(fy - ay * scale),
            max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale)))


def _outside(cam, box):
    x, y, w, h = box
    return x >= cam.w or y >= cam.h or x + w <= 0 or y + h <= 0


def _walk_end(cam, route, lane, obstacle, height, sprite, anchor, direction):
    center = (route.start + route.end) / 2
    along = center
    for i in range(1, 33):
        candidate = center + direction * i * cam.dist / 16
        x, y = _lane_point(route, candidate, lane, obstacle)
        depth, fx, fy, pixel_h = _walker_projection(cam, x, y, height)
        if _outside(cam, _sprite_box(sprite, anchor, fx, fy, pixel_h / sprite.height)):
            return candidate, 0.0
        if i > 1 and depth < cam.dist / 4:
            break
        along = candidate

    # A route aimed at a visible vanishing point might never clear the frame.
    # At the end of its straight segment, continue toward a side at fixed depth.
    x, y = _lane_point(route, along, lane, obstacle)
    depth, fx, fy, pixel_h = _walker_projection(cam, x, y, height)
    left, _, width, _ = _sprite_box(sprite, anchor, fx, fy, pixel_h / sprite.height)
    right = cam.ca if route.along_x else -cam.sa
    if abs(right) <= 1e-9:
        right = 1.0
    shift = cam.w - left + 2 if direction * right > 0 else -left - width - 2
    return along, shift * max(depth, cam.dist / 4) / cam.focal


def make_crowd(p, routes: tuple[WalkRoute, ...], cam, obstacle: Obstacle | None = None):
    """Build a reproducible crowd whose complete sprites enter and exit off-screen."""
    if not routes:
        return []
    rng = random.Random(p.crowd_seed)
    angle = math.radians(p.angle)
    sin_a, cos_a = math.sin(angle), math.cos(angle)
    route_order = rng.sample(routes, len(routes))
    people = []
    for i in range(int(p.crowd_count)):
        route = route_order[i % len(route_order)]
        sign = rng.choice((-1, 1))
        right, away = (cos_a, sin_a) if route.along_x else (-sin_a, cos_a)
        frames, anchor = _build_frames(
            "toward" if sign * away < -1e-9 else "away",
            random_look(rng), female=rng.random() < 0.5,
            mirror=sign * right > 1e-9)
        lane = rng.uniform(-0.75, 0.75) * route.half_depth
        progress = rng.random()
        rate = p.crowd_speed * rng.uniform(0.8, 1.25)
        phase = rng.uniform(0, FRAME_COUNT)
        height = p.person_size * rng.uniform(1 - p.size_var, 1 + p.size_var)
        start, entry = _walk_end(cam, route, lane, obstacle, height, frames[0], anchor, -1)
        end, exit = _walk_end(cam, route, lane, obstacle, height, frames[0], anchor, 1)
        people.append(Person(
            route=WalkRoute(route.along_x, start, end, route.half_depth),
            lane=lane,
            offset=progress * (end - start + abs(entry) + abs(exit)),
            dir_sign=sign,
            rate=rate,
            phase=phase,
            height=height,
            frames=frames,
            anchor=anchor,
            entry=entry,
            exit=exit,
            right=(cos_a, -sin_a),
        ))
    return people


# ---------------------------------------------------------------- rendering


def _project(img, cam, x, y, height):
    """(depth, foot x, foot y, pixel height) of an upright figure, or None."""
    feet = cam.to_cam(x, y)
    head = cam.to_cam(x, y, height)
    if feet[2] < cam.near or head[2] < cam.near:
        return None
    fx, fy = cam.project(feet)
    _, hy = cam.project(head)
    pixel_h = fy - hy
    if pixel_h < 2 or pixel_h > img.height * 3:
        return None
    return feet[2], fx, fy, pixel_h


def draw_frame(img, cam, people, t, dog=None):
    """Composite the crowd (and optional dog) at frame t onto img (in place)."""
    obstacle = dog.obstacle() if dog is not None else None
    visible = []
    for person in people:
        x, y = person.ground_pose(t, obstacle)
        depth, fx, fy, pixel_h = _walker_projection(cam, x, y, person.height)
        sprite = person.frames[int(person.phase + person.rate * t) % FRAME_COUNT]
        visible.append((depth, sprite, person.anchor, fx, fy, pixel_h / sprite.height))
    if dog is not None:
        proj = _project(img, cam, dog.x, dog.y, dog.height)
        if proj is not None:
            depth, fx, fy, pixel_h = proj
            sprite, anchor = dog.sprite(cam, people, t)
            visible.append((depth, sprite, anchor, fx, fy, pixel_h / dog.WORK_HEIGHT))
    visible.sort(key=lambda item: -item[0])  # far to near
    for _, sprite, anchor, fx, fy, scale in visible:
        box = _sprite_box(sprite, anchor, fx, fy, scale)
        if _outside(cam, box):
            continue
        px, py, w, h = box
        scaled = sprite.resize((w, h), Image.LANCZOS)
        img.paste(scaled, (px, py), scaled.getchannel("A"))
    return img
