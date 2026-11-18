"""Randomized pedestrian crowds for the crosswalk generator.

Loads the layered 12-frame walk cycles from Animations/ (legs, arms, head,
hair_male / hair_female), recolors each layer per person while preserving the
crayon texture, and renders the crowd with perspective scaling through the
generator's camera.
"""

import colorsys
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

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
PANTS_COLORS = [
    (45, 62, 92), (72, 101, 140), (52, 54, 58), (125, 126, 122),
    (166, 145, 108), (105, 76, 57), (91, 96, 65),
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
    shirt, _ = _clothing_color(rng)
    pants = _jitter(rng, rng.choice(PANTS_COLORS), 8)
    return {
        "legs": pants,
        "arms": shirt,
        "head": _jitter(rng, rng.choice(SKIN_TONES), 8),
        "hair": _jitter(rng, rng.choice(HAIR_COLORS), 10),
    }


class _PersonFrames(Sequence):
    def __init__(self, data, order, mirror):
        self.data = data
        self.order = order
        self.mirror = mirror
        self.cache = {}

    def __len__(self):
        return FRAME_COUNT

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = range(len(self))[index]
        if index in self.cache:
            return self.cache[index]
        width, height = self.data["size"]
        comp = Image.new("RGBA", (width, height))
        for layer, color in self.order:
            norm, alpha = self.data["layers"][layer][index]
            rgb = np.clip(norm[..., None].astype(np.float32) * np.asarray(color, dtype=np.float32), 0, 255).astype(np.uint8)
            comp = Image.alpha_composite(comp, Image.fromarray(np.dstack([rgb, alpha])))
        if self.mirror:
            comp = comp.transpose(Image.FLIP_LEFT_RIGHT)
        self.cache[index] = comp
        return comp


def _build_frames(direction, colors, female, mirror):
    """Recolor and cache each person's sprites only when needed."""
    data = load_assets()[direction]
    width, _ = data["size"]
    hair = "hair_female" if female else "hair_male"
    order = tuple((layer, tuple(color)) for layer, color in (
        ("legs", colors["legs"]), ("arms", colors["arms"]),
        ("head", colors["head"]), (hair, colors["hair"])))
    ax, ay = data["anchor"]
    if mirror:
        ax = width - ax
    return _PersonFrames(data, order, mirror), (ax, ay)


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
    frames: Sequence  # 12 lazily composited RGBA sprites
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


def _draw_shadow(mask, cam, sprite, anchor, sprite_height, x, y, height,
                 depth_shift, style, seed, drawing):
    """Project a contact blob or the sprite's alpha onto the ground."""
    alpha = sprite.getchannel("A")
    if style == "contact":
        if not alpha.getbbox():
            return
        alpha = Image.new("L", (64, 64))
        rng = random.Random(f"shadow:{seed}:{drawing}")
        points = []
        for i in range(24):
            angle = 2 * math.pi * i / 24
            radius = rng.uniform(0.94, 1.06)
            points.append((32 + 11 * radius * math.cos(angle),
                           32 + 28 * radius * math.sin(angle)))
        ImageDraw.Draw(alpha).polygon(points, fill=255)
        anchor, sprite_height = (32, 32), 64
    direction = (math.sqrt(0.5), math.sqrt(0.5))
    side = (-direction[1], direction[0])
    # Keep increasing sprite x pointed toward camera-right.
    if side[0] * cam.ca - side[1] * cam.sa < 0:
        side = (-side[0], -side[1])
    unit = height / sprite_height
    length = 0.85 if style == "cast" else 0.24
    ax, ay = anchor
    origin = (x - side[0] * ax * unit + direction[0] * ay * unit * length,
              y - side[1] * ax * unit + direction[1] * ay * unit * length)
    base = np.array(cam.to_cam(*origin))
    right = np.array(cam.to_cam(origin[0] + side[0] * unit,
                                origin[1] + side[1] * unit)) - base
    down = np.array(cam.to_cam(origin[0] - direction[0] * unit * length,
                               origin[1] - direction[1] * unit * length)) - base
    base[2] += depth_shift
    camera_matrix = np.column_stack((right, down, base))
    corners = [camera_matrix @ (u, v, 1) for u, v in
               ((0, 0), (alpha.width, 0), (alpha.width, alpha.height), (0, alpha.height))]
    clipped = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        if a[2] >= cam.near:
            clipped.append(a)
        if (a[2] >= cam.near) != (b[2] >= cam.near):
            clipped.append(a + (b - a) * ((cam.near - a[2]) / (b[2] - a[2])))
    if not clipped:
        return
    polygon = [cam.project(point) for point in clipped]
    left = max(0, math.floor(min(point[0] for point in polygon)))
    top = max(0, math.floor(min(point[1] for point in polygon)))
    right_edge = min(mask.width, math.ceil(max(point[0] for point in polygon)))
    bottom = min(mask.height, math.ceil(max(point[1] for point in polygon)))
    if left >= right_edge or top >= bottom:
        return

    projection = np.array(((cam.focal, 0, cam.w / 2),
                           (0, -cam.focal, cam.h / 2),
                           (0, 0, 1))) @ camera_matrix
    inverse = np.linalg.inv(projection) @ np.array(((1, 0, left), (0, 1, top), (0, 0, 1)))
    inverse /= inverse[2, 2]
    coefficients = tuple(inverse.flat[:8])
    size = (right_edge - left, bottom - top)
    shadow = alpha.transform(
        size, Image.Transform.PERSPECTIVE, coefficients, resample=Image.Resampling.BICUBIC)
    clip = Image.new("L", size)
    ImageDraw.Draw(clip).polygon([(px - left, py - top) for px, py in polygon], fill=255)
    opacity = random.Random(f"shadow:{seed}:{drawing}").uniform(0.32, 0.38)
    shadow = ImageChops.multiply(shadow, clip).point(lambda value: round(value * opacity))
    region = mask.crop((left, top, right_edge, bottom))
    mask.paste(ImageChops.lighter(region, shadow), (left, top))


def draw_frame(img, cam, people, t, dog=None, *, shadows=False,
               shadow_style="contact", shadow_hold=1):
    """Composite the crowd (and optional dog) at frame t onto img (in place)."""
    if shadow_style not in ("contact", "cast"):
        raise ValueError("Shadow style must be 'contact' or 'cast'.")
    obstacle = dog.obstacle() if dog is not None else None
    visible = []
    shadow_poses = []
    for index, person in enumerate(people):
        x, y = person.ground_pose(t, obstacle)
        depth, fx, fy, pixel_h = _walker_projection(cam, x, y, person.height)
        sprite = person.frames[int(person.phase + person.rate * t) % FRAME_COUNT]
        visible.append((depth, sprite, person.anchor, fx, fy, pixel_h / sprite.height))
        if shadows:
            shadow_poses.append((sprite, person.anchor, sprite.height, x, y, person.height,
                                 max(depth, cam.dist / 4) - depth, f"person:{index}"))
    if dog is not None:
        proj = _project(img, cam, dog.x, dog.y, dog.height)
        if proj is not None:
            depth, fx, fy, pixel_h = proj
            sprite, anchor = dog.sprite(cam, people, t)
            visible.append((depth, sprite, anchor, fx, fy, pixel_h / dog.WORK_HEIGHT))
            if shadows:
                shadow_poses.append((sprite, anchor, dog.WORK_HEIGHT, dog.x, dog.y,
                                     dog.height, 0, "dog"))
    if shadow_poses:
        mask = Image.new("L", img.size)
        drawing = t // max(1, shadow_hold)
        for sprite, anchor, sprite_height, x, y, height, depth_shift, seed in shadow_poses:
            _draw_shadow(mask, cam, sprite, anchor, sprite_height, x, y, height, depth_shift,
                         shadow_style, seed, drawing)
        color = (25, 25, 25, 255) if img.mode == "RGBA" else (25, 25, 25)
        img.paste(color, mask=mask)
    visible.sort(key=lambda item: -item[0])  # far to near
    for _, sprite, anchor, fx, fy, scale in visible:
        box = _sprite_box(sprite, anchor, fx, fy, scale)
        if _outside(cam, box):
            continue
        px, py, w, h = box
        scaled = sprite.resize((w, h), Image.LANCZOS)
        img.paste(scaled, (px, py), scaled.getchannel("A"))
    return img
