"""Randomized pedestrian crowds for the crosswalk generator.

Loads the layered 12-frame walk cycles from Animations/ (legs, arms, head,
hair_male / hair_female), recolors each layer per person while preserving the
crayon texture, and renders the crowd with perspective scaling through the
generator's camera.
"""

import colorsys
import math
import random
import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
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
        assets[direction] = _load_layers({
            layer: [ANIM_DIR / folder / layer / f"{prefix}_{layer}{i:04d}.png"
                    for i in range(1, FRAME_COUNT + 1)]
            for layer in BODY_LAYERS + HAIR_LAYERS
        })
    _ASSETS = assets
    return assets


@lru_cache(maxsize=4)
def load_clip(folder: str):
    root = Path(folder)
    paths = {}
    expected = None
    for layer in BODY_LAYERS + HAIR_LAYERS:
        directory = root / layer
        if not directory.is_dir():
            raise ValueError(f"Missing animation layer folder: {directory}")
        numbered = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() != ".png":
                continue
            match = re.search(r"([0-9]+)$", path.stem)
            if match is None or int(match[1]) in numbered:
                raise ValueError(f"Animation frames must have unique trailing numbers: {path}")
            numbered[int(match[1])] = path
        indices = sorted(numbered)
        if (not indices or indices[-1] - indices[0] + 1 != len(indices)
                or (expected is not None and indices != expected)):
            raise ValueError(f"Expected matching, consecutive numbered PNG frames in {directory}")
        expected = indices
        paths[layer] = [numbered[index] for index in indices]
    return _load_layers(paths, aligned=True)


@lru_cache(maxsize=4)
def load_side_walk(folder: str):
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"Walk-cycle folder is missing: {folder}")
    numbered = {}
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        match = re.search(r"([0-9]+)$", path.stem)
        if match is None or int(match[1]) in numbered:
            raise ValueError(f"Walk-cycle frames must have unique trailing numbers: {path}")
        numbered[int(match[1])] = path
    indices = sorted(numbered)
    if not indices or indices[-1] - indices[0] + 1 != len(indices):
        raise ValueError(f"Expected consecutive numbered PNG frames in {root}")
    frames, bbox, size = [], None, None
    for index in indices:
        path = numbered[index]
        try:
            with Image.open(path, formats=("PNG",)) as image:
                frame = image.convert("RGBA")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ValueError(f"Cannot read walk-cycle frame {path}: {error}") from error
        if size is not None and frame.size != size:
            raise ValueError(f"Walk-cycle frames must share one canvas size: {path}")
        size = frame.size
        frames.append(frame)
        if (bounds := frame.getbbox()) is not None:
            bbox = _union(bbox, bounds)
    if bbox is None:
        raise ValueError(f"Walk-cycle frames are fully transparent: {folder}")
    left, top, right, bottom = bbox
    width = max(1, round((right - left) * WORK_HEIGHT / (bottom - top)))
    frames = tuple(frame.crop(bbox).resize((width, WORK_HEIGHT), Image.LANCZOS)
                   for frame in frames)
    mirrored = tuple(frame.transpose(Image.FLIP_LEFT_RIGHT) for frame in frames)
    return (frames, mirrored), (width / 2, WORK_HEIGHT)


def _load_layers(paths, *, aligned=False, reference_height=None):
    raw = {}
    bbox = legs_bbox = size = None
    for layer, files in paths.items():
        frames = []
        layer_bbox = None
        for path in files:
            try:
                with Image.open(path, formats=("PNG",)) as image:
                    frame = image.convert("RGBA")
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                raise ValueError(f"Cannot read animation frame {path}: {error}") from error
            if aligned and size is not None and frame.size != size:
                raise ValueError(f"Animation layers must share one canvas size: {path}")
            size = frame.size
            frames.append(frame)
            if (bounds := frame.getbbox()) is not None:
                layer_bbox = _union(layer_bbox, bounds)
        if layer_bbox is None:
            raise ValueError(f"Animation layer is fully transparent: {files[0].parent}")
        bbox = _union(bbox, layer_bbox)
        if layer == "legs":
            legs_bbox = layer_bbox
        raw[layer] = frames
    left, top, right, bottom = bbox
    raw_height = bottom - top
    if reference_height is not None:
        top = bottom - reference_height
        bbox = left, top, right, bottom
    scale = WORK_HEIGHT / (bottom - top)
    width = max(1, round((right - left) * scale))
    layers = {}
    for layer, frames in raw.items():
        normalized = []
        values = []
        for frame in frames:
            a = np.asarray(frame.crop(bbox).resize((width, WORK_HEIGHT), Image.LANCZOS))
            brightness = a[..., :3].max(-1)
            alpha = a[..., 3].copy()
            normalized.append((brightness, alpha))
            if (mask := alpha > 64).any():
                values.append(brightness[mask])
        # Map a typical fully-colored stroke to the chosen color, retaining texture.
        reference = max(float(np.percentile(np.concatenate(values), 95)), 1.0) if values else 1.0
        layers[layer] = [(np.divide(v, reference, dtype=np.float32).astype(np.float16), alpha)
                         for v, alpha in normalized]
    anchor = (((legs_bbox[0] + legs_bbox[2]) / 2 - left) * scale,
              (legs_bbox[3] - top) * scale)
    return {"layers": layers, "size": (width, WORK_HEIGHT), "anchor": anchor,
            "raw_height": raw_height}


@lru_cache(maxsize=1)
def load_petting():
    root = ANIM_DIR / "petting"
    return _load_layers({
        layer: [root / layer / f"petting_{layer}{i:04d}.png" for i in (1, 2)]
        for layer in BODY_LAYERS + ("hair_male",)
    }, aligned=True, reference_height=load_assets()["away"]["raw_height"])


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
        return len(self.data["layers"]["legs"])

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
    return _color_frames(load_assets()[direction], colors, female, mirror)


def _color_frames(data, colors, female, mirror):
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
class ClipBlock:
    folder: str
    start: int
    end: int


@dataclass(frozen=True)
class CustomPedestrian:
    id: str
    route: tuple[tuple[float, float], ...] = ()
    colors: tuple[str, str, str, str] = ("#495c72", "#bc603b", "#f0c8aa", "#3c2819")
    female: bool = False
    speed: float = 1.5
    departure: int = 0
    clips: tuple[ClipBlock, ...] = ()
    height: float | None = None  # None inherits the shot's crowd person size
    walk_duration: int | None = None  # Inclusive frame count; None preserves legacy speed.
    pet_dog: bool = False
    pet_seconds: float = 2.0
    pet_offset: tuple[float, float] = (0.0, 0.0)
    side_walk_folder: str | None = None
    side_walk_stride: float = STRIDE_FACTOR

    def walk_end(self, fps):
        if self.walk_duration is not None:
            return self.departure + self.walk_duration - 1
        return self.departure + max(1, math.ceil(self.route_length() * fps / self.speed))

    def route_length(self):
        return route_distances(self.route)[-1]

    def walk_speed(self, fps):
        if self.walk_duration is None or len(self.route) < 2:
            return self.speed
        return self.route_length() * fps / (self.walk_duration - 1)


def route_distances(route):
    distances = [0.0]
    for start, end in zip(route, route[1:]):
        distances.append(distances[-1] + math.dist(start, end))
    return distances


def _route_pose(route, distances, distance):
    distance = min(distances[-1], max(0.0, distance))
    segment = min(bisect_right(distances, distance) - 1, len(route) - 2)
    x, y = route[segment]
    nx, ny = route[segment + 1]
    length = distances[segment + 1] - distances[segment]
    fraction = (distance - distances[segment]) / length
    return x + (nx - x) * fraction, y + (ny - y) * fraction, nx - x, ny - y, distance


class DogInteraction:
    """Resolved, random-access choreography shared by all rendering paths."""

    def __init__(self, spec, height, fps, cam, dog_height):
        self.spec, self.height, self.fps = spec, height, fps
        self.stop = (-cam.ca * height * .36 - cam.sa * height * .1 + spec.pet_offset[0],
                     cam.sa * height * .36 - cam.ca * height * .1 + spec.pet_offset[1])
        route = []
        for point in spec.route:
            if not route or math.dist(route[-1], point) > 1e-8:
                route.append(point)
        if len(route) < 2:
            raise ValueError("The interaction route needs two distinct points.")
        if math.dist(route[-1], self.stop) > 1e-8:
            route.append(self.stop)
        else:
            route[-1] = self.stop
        # Extend backwards along the first leg until both complete sprites clear.
        start, following = route[:2]
        length = math.dist(start, following)
        ux, uy = ((start[0] - following[0]) / length,
                  (start[1] - following[1]) / length)
        step = max(height, dog_height, cam.dist / 8)
        entry = start
        for index in range(129):
            entry = (start[0] + ux * index * step, start[1] + uy * index * step)
            depth, sx, sy, ph = _walker_projection(cam, *entry, max(height, dog_height))
            if depth > cam.near and _outside(cam, (sx - ph, sy - ph, 2 * ph, ph)):
                break
        else:
            # A vanishing-point route needs a camera-horizontal hidden extension.
            depth, sx, sy, ph = _walker_projection(cam, *start, max(height, dog_height))
            shift = (-sx - 2 * ph) * max(depth, cam.dist / 4) / cam.focal
            entry = start[0] + cam.ca * shift, start[1] - cam.sa * shift
        if math.dist(entry, start) > 1e-8:
            route.insert(0, entry)
        self.route = tuple(route)
        self.distances = route_distances(self.route)
        self.duration = (spec.walk_duration if spec.walk_duration is not None else
                         max(2, math.ceil(self.distances[-1] * fps / spec.speed) + 1))
        self.speed = self.distances[-1] * fps / (self.duration - 1)
        self.pet_start = spec.departure + self.duration - 1
        self.pet_frames = max(1, round(spec.pet_seconds * fps))
        self.return_start = self.pet_start + self.pet_frames
        self.return_route = tuple(reversed(self.route))
        self.return_distances = route_distances(self.return_route)
        dog_route = [(0.0, 0.0)]
        dog_route.extend(point for point in self.return_route
                         if math.dist(point, dog_route[-1]) > 1e-8)
        self.dog_route = tuple(dog_route)
        self.dog_distances = route_distances(self.dog_route)
        self.follow_gap = height * .65
        self.dog_delay = max(0.0, self.follow_gap - math.dist((0, 0), self.stop))
        self.end = self.return_start + math.ceil(
            (self.dog_distances[-1] + self.dog_delay) * fps / self.speed)
        self.obstacles = None

    def petting(self, frame):
        return self.pet_start <= frame < self.return_start

    def pose(self, frame):
        if frame < self.pet_start:
            return _route_pose(self.route, self.distances,
                               (frame - self.spec.departure) * self.speed / self.fps)
        if self.petting(frame):
            return (*self.stop, 1.0, 0.0, self.distances[-1])
        pose = _route_pose(self.return_route, self.return_distances,
                           (frame - self.return_start) * self.speed / self.fps)
        return (*pose[:4], self.distances[-1] + pose[4])

    def dog_pose(self, frame):
        distance = max(0.0, (frame - self.return_start) * self.speed / self.fps
                       - self.dog_delay)
        return _route_pose(self.dog_route, self.dog_distances, distance)


class CustomWalker:
    def __init__(self, spec, height, fps):
        self.spec = spec
        self.height = height
        self.fps = fps
        self.distances = route_distances(spec.route)
        self.colors = {part: tuple(bytes.fromhex(color[1:]))
                       for part, color in zip((*BODY_LAYERS, "hair"), spec.colors)}
        self.frames = {}
        self.interaction = None
        self.pet_hold = 1

    def ground_pose(self, frame, _obstacle=None):
        return self.pose(frame)[:2]

    def pose(self, frame):
        if self.interaction is not None:
            return self.interaction.pose(frame)
        spec = self.spec
        if len(spec.route) == 1:
            return (*spec.route[0], 0.0, 0.0, 0.0)
        elapsed = max(0, frame - spec.departure)
        if spec.walk_duration is None:
            distance = min(self.distances[-1], elapsed * spec.speed / self.fps)
        else:
            distance = self.distances[-1] * min(1.0, elapsed / (spec.walk_duration - 1))
        segment = min(bisect_right(self.distances, distance) - 1, len(spec.route) - 2)
        x, y = spec.route[segment]
        nx, ny = spec.route[segment + 1]
        length = self.distances[segment + 1] - self.distances[segment]
        fraction = (distance - self.distances[segment]) / length
        return x + (nx - x) * fraction, y + (ny - y) * fraction, nx - x, ny - y, distance

    def sprite(self, cam, frame, dx, dy, distance, *, allow_missing=False):
        if self.spec.side_walk_folder is not None:
            if "side_walk" not in self.frames:
                self.frames["side_walk"] = load_side_walk(self.spec.side_walk_folder)
            variants, anchor = self.frames["side_walk"]
            if len(self.spec.route) == 1:
                frames = variants[0]
                return frames[frame % len(frames)], anchor
            right = dx * cam.ca - dy * cam.sa
            if abs(right) <= 1e-9:
                segment = min(bisect_right(self.distances, distance) - 1, len(self.spec.route) - 2)
                for i in range(segment - 1, -1, -1):
                    start, end = self.spec.route[i:i + 2]
                    right = (end[0] - start[0]) * cam.ca - (end[1] - start[1]) * cam.sa
                    if abs(right) > 1e-9:
                        break
            frames = variants[right > 1e-9]
            moving = frame >= self.spec.departure and distance < self.distances[-1]
            index = int(distance / (self.spec.side_walk_stride * self.height) * len(frames)) if moving else 0
            return frames[index % len(frames)], anchor
        if self.interaction is not None and self.interaction.petting(frame):
            if "petting" not in self.frames:
                self.frames["petting"] = _color_frames(load_petting(), self.colors, False, False)
            frames, anchor = self.frames["petting"]
            return frames[((frame - self.interaction.pet_start) // self.pet_hold) % len(frames)], anchor
        block = next((clip for clip in self.spec.clips if clip.start <= frame <= clip.end), None)
        if block is not None and (not allow_missing or Path(block.folder).is_dir()):
            key = block.folder
            if key not in self.frames:
                self.frames[key] = _color_frames(load_clip(key), self.colors, self.spec.female, False)
            frames, anchor = self.frames[key]
            return frames[(frame - block.start) % len(frames)], anchor
        away = dx * cam.sa + dy * cam.ca
        right = dx * cam.ca - dy * cam.sa
        key = "toward" if away < -1e-9 else "away", right > 1e-9
        if key not in self.frames:
            self.frames[key] = _build_frames(key[0], self.colors, self.spec.female, key[1])
        frames, anchor = self.frames[key]
        moving = frame >= self.spec.departure and (
            self.interaction is not None or distance < self.distances[-1])
        index = int(distance / (STRIDE_FACTOR * self.height) * len(frames)) if moving else 0
        return frames[index % len(frames)], anchor


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


class InteractionObstacles:
    def __init__(self, interaction, dog):
        self.interaction, self.dog = interaction, dog
        self.history = {}

    def at(self, frame):
        x, y = self.interaction.pose(frame)[:2]
        dx, dy = self.interaction.dog_pose(frame)[:2]
        return (Obstacle(x, y, self.interaction.height * .35),
                Obstacle(dx, dy, max(self.dog.clearance, self.dog.height * .5)))

    def _active(self, route, along, frame):
        active = []
        for obstacle in self.at(frame):
            center, side = ((obstacle.x, obstacle.y) if route.along_x
                            else (obstacle.y, obstacle.x))
            radius = obstacle.radius
            distance = abs(along - center)
            if distance >= 3 * radius:
                continue
            u = min(1.0, (3 * radius - distance) / (2 * radius))
            active.append((side, radius, u * u * (3 - 2 * u)))
        return active

    def offset(self, person, frame):
        # Retain the passing side through an encounter; choosing the nearest side
        # independently each frame makes walkers jump across moving obstacles.
        history = self.history.setdefault(id(person), [])
        for t in range(len(history), max(0, frame) + 1):
            x, y = person.ground_pose(t)
            along = x if person.route.along_x else y
            active = self._active(person.route, along, t)
            sign = history[-1][0] if history else 0
            if not active:
                sign, offset = 0, 0.0
            else:
                if not sign:
                    center = max(active, key=lambda item: item[2])[0]
                    sign = 1 if person.lane >= center else -1
                offset = sign * max(
                    max(0, sign * (side - person.lane) + radius) * influence
                    for side, radius, influence in active)
            history.append((sign, offset))
        return history[max(0, frame)][1]


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
    direction: str = "away"
    mirror: bool = False
    replacement_seed: str = "0"
    replacement_frames: dict[int, Sequence] = field(default_factory=dict)

    def appearance_generation(self, t):
        length = abs(self.entry) + self.route.end - self.route.start + abs(self.exit)
        step = self.rate / FRAME_COUNT * STRIDE_FACTOR * self.height
        period = length + 2 * step
        return abs(math.floor((self.offset + self.dir_sign * step * t) / period))

    def frames_for_generation(self, generation):
        if generation <= 0:
            return self.frames
        if generation not in self.replacement_frames:
            rng = random.Random(f"{self.replacement_seed}:{generation}")
            female = rng.random() < 0.5
            frames, _ = _build_frames(
                self.direction, random_look(rng), female=female, mirror=self.mirror)
            self.replacement_frames[generation] = frames
        return self.replacement_frames[generation]

    def ground_pose(self, t, obstacle: Obstacle | InteractionObstacles | None = None):
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
        if isinstance(obstacle, InteractionObstacles):
            lane = self.lane + obstacle.offset(self, t)
            x, y = _lane_point(route, along, lane, None)
        else:
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
        direction = "toward" if sign * away < -1e-9 else "away"
        mirror = sign * right > 1e-9
        look = random_look(rng)
        female = rng.random() < 0.5
        frames, anchor = _build_frames(
            direction, look, female=female, mirror=mirror)
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
            direction=direction,
            mirror=mirror,
            replacement_seed=f"{p.crowd_seed}:{i}",
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
                 depth_shift, style, seed, drawing, pixel_height):
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
    unit = height / sprite_height
    side_unit = unit
    sampling = 1
    if style == "cast":
        side = (cam.ca, -cam.sa)
        forward = (cam.sa, cam.ca)
        along = direction[0] * side[0] + direction[1] * side[1]
        across = direction[0] * forward[0] + direction[1] * forward[1]
        # A light exactly parallel to the sprite plane has a singular projection.
        if abs(across) < 0.05:
            across = math.copysign(0.05, across)
            norm = math.hypot(along, across)
            direction = ((along * side[0] + across * forward[0]) / norm,
                         (along * side[1] + across * forward[1]) / norm)
        if abs(across) < 0.1:
            sampling = 8
        depth = cam.to_cam(x, y)[2] + depth_shift
        side_unit = pixel_height * depth / (sprite_height * cam.focal)
    else:
        side = (-direction[1], direction[0])
        if side[0] * cam.ca - side[1] * cam.sa < 0:
            side = (-side[0], -side[1])
    length = 0.85 if style == "cast" else 0.24
    ax, ay = anchor
    origin = (x - side[0] * ax * side_unit + direction[0] * ay * unit * length,
              y - side[1] * ax * side_unit + direction[1] * ay * unit * length)
    base = np.array(cam.to_cam(*origin))
    right = np.array(cam.to_cam(origin[0] + side[0] * side_unit,
                                origin[1] + side[1] * side_unit)) - base
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
    size = (right_edge - left, bottom - top)
    sampling = min(sampling, max(1, math.isqrt(4_000_000 // (size[0] * size[1]))))
    inverse = np.linalg.inv(projection) @ np.array(
        ((1 / sampling, 0, left), (0, 1 / sampling, top), (0, 0, 1)))
    inverse /= inverse[2, 2]
    coefficients = tuple(inverse.flat[:8])
    sample_size = (size[0] * sampling, size[1] * sampling)
    shadow = alpha.transform(
        sample_size, Image.Transform.PERSPECTIVE, coefficients, resample=Image.Resampling.BICUBIC)
    clip = Image.new("L", sample_size)
    ImageDraw.Draw(clip).polygon([((px - left) * sampling, (py - top) * sampling)
                                  for px, py in polygon], fill=255)
    opacity = random.Random(f"shadow:{seed}:{drawing}").uniform(0.32, 0.38)
    shadow = ImageChops.multiply(shadow, clip).point(lambda value: round(value * opacity))
    if sampling > 1:
        shadow = shadow.resize(size, Image.Resampling.LANCZOS)
    region = mask.crop((left, top, right_edge, bottom))
    mask.paste(ImageChops.lighter(region, shadow), (left, top))


def draw_frame(img, cam, people, t, dog=None, *, shadows=False,
               shadow_style="contact", shadow_hold=1, custom=(), allow_missing=False,
               loop_pedestrians=True):
    """Composite the crowd (and optional dog) at frame t onto img (in place)."""
    if shadow_style not in ("contact", "cast"):
        raise ValueError("Shadow style must be 'contact' or 'cast'.")
    obstacle = dog.obstacle() if dog is not None else None
    interaction = next((walker.interaction for walker in custom
                        if walker.interaction is not None), None)
    if interaction is not None and dog is not None:
        if interaction.obstacles is None:
            interaction.obstacles = InteractionObstacles(interaction, dog)
        obstacle = interaction.obstacles
    visible = []
    shadow_poses = []
    gaze_people = list(people) if dog is not None else []
    for index, person in enumerate(people):
        x, y = person.ground_pose(t, obstacle)
        depth, fx, fy, pixel_h = _walker_projection(cam, x, y, person.height)
        frames = (person.frames if loop_pedestrians else
                  person.frames_for_generation(person.appearance_generation(t)))
        sprite = frames[int(person.phase + person.rate * t) % FRAME_COUNT]
        visible.append((depth, sprite, person.anchor, fx, fy, pixel_h / sprite.height))
        if shadows:
            shadow_poses.append((sprite, person.anchor, sprite.height, x, y, person.height,
                                 max(depth, cam.dist / 4) - depth, f"person:{index}", pixel_h))
    for walker in custom:
        x, y, dx, dy, distance = walker.pose(t)
        if (allow_missing and walker.spec.side_walk_folder is not None
                and not Path(walker.spec.side_walk_folder).is_dir()):
            continue
        if dog is not None:
            gaze_people.append(walker)
        if cam.to_cam(x, y)[2] <= cam.near:
            continue
        depth, fx, fy, pixel_h = _walker_projection(cam, x, y, walker.height)
        sprite, anchor = walker.sprite(cam, t, dx, dy, distance, allow_missing=allow_missing)
        visible.append((depth, sprite, anchor, fx, fy, pixel_h / sprite.height))
        if shadows:
            shadow_poses.append((sprite, anchor, sprite.height, x, y, walker.height,
                                 max(depth, cam.dist / 4) - depth, f"custom:{walker.spec.id}", pixel_h))
    if dog is not None:
        x, y = interaction.dog_pose(t)[:2] if interaction else (dog.x, dog.y)
        proj = _project(img, cam, x, y, dog.height)
        if proj is not None:
            depth, fx, fy, pixel_h = proj
            sprite, anchor = (dog.interaction_sprite(cam, gaze_people, t, interaction, obstacle)
                              if interaction else dog.sprite(cam, gaze_people, t))
            visible.append((depth, sprite, anchor, fx, fy, pixel_h / dog.WORK_HEIGHT))
            if shadows:
                shadow_poses.append((sprite, anchor, dog.WORK_HEIGHT, x, y,
                                     dog.height, 0, "dog", pixel_h))
    if shadow_poses:
        mask = Image.new("L", img.size)
        drawing = t // max(1, shadow_hold)
        for sprite, anchor, sprite_height, x, y, height, depth_shift, seed, pixel_h in shadow_poses:
            _draw_shadow(mask, cam, sprite, anchor, sprite_height, x, y, height, depth_shift,
                         shadow_style, seed, drawing, pixel_h)
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
