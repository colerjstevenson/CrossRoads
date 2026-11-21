"""3/4 overhead crosswalk image generator.

Run:  python crosswalk_generator.py

Stripe wobble controls both stripe jitter and the motion of the road
texture. Each can be held on its first drawing independently.
Both use the Dog tab's frame hold, even when the dog is hidden.
The texture snaps to a new angle each drawing. Background zoom changes its
scale without affecting the camera. Below 1.0, the road texture is tiled.
Save PNG renders the displayed animation frame at the selected output size.
Pedestrians enter and exit outside the frame. GIFs with pedestrians play once
so restarting a short capture doesn't reset walkers in view.
"""

import json
import math
import multiprocessing
import os
import queue
import random
import re
import tempfile
import threading
import tkinter as tk
import uuid
from dataclasses import asdict, dataclass, field, replace
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageTk

import crowd
import dog as dog_mod

HFOV_DEG = 40.0  # horizontal field of view of the virtual camera
MAX_BARS_PER_BAND = 4000
ROAD_TEXTURE = Path(__file__).with_name("BG.png")
BACKGROUND_TILES = 5
MIN_BACKGROUND_ZOOM = 1.0 / BACKGROUND_TILES
MAX_BACKGROUND_ZOOM = 2.0
MIN_PERSON_SIZE, MAX_PERSON_SIZE = 0.5, 4.0
BACKGROUND_TURN_DEG = 110
DEFAULTS_PATH = Path.home() / ".crossroads" / "defaults.json"
_PREVIEW_RENDERER = None


@dataclass
class _JobResult:
    value: object = None
    progress: tuple[int, str] | None = None
    error: Exception | None = None


def _put_progress(progress, value):
    while True:
        try:
            progress.put_nowait(value)
            return
        except queue.Full:
            try:
                progress.get_nowait()
            except queue.Empty:
                pass


def _initialize_worker(cancelled, progress):
    global _WORKER_CANCELLED, _WORKER_PROGRESS
    _WORKER_CANCELLED, _WORKER_PROGRESS = cancelled, progress


def _run_worker(function, token):
    return _run_job(function, _WORKER_CANCELLED, _WORKER_PROGRESS, token)


def _run_job(function, cancelled, progress, token):
    latest = None

    def report(value):
        nonlocal latest
        latest = value
        _put_progress(progress, (token, value))

    try:
        return _JobResult(function(cancelled, report), latest)
    except Exception as error:
        return _JobResult(progress=latest, error=error)


class _BackgroundWorker:
    def __init__(self):
        context = multiprocessing.get_context("spawn")
        self.cancelled = context.Event()
        self.progress = context.Queue(maxsize=1)
        self.pool = ProcessPoolExecutor(
            max_workers=1, mp_context=context, initializer=_initialize_worker,
            initargs=(self.cancelled, self.progress))

    def submit(self, function, token):
        return self.pool.submit(_run_worker, function, token)

    def shutdown(self, *, wait=False, cancel_futures=True):
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        self.progress.close()


@dataclass
class Params:
    width: int = 1080
    height: int = 1080
    mode: str = "single"  # "single" or "cross"
    angle: float = 0.0  # rotation of the crosswalk in the image (degrees)
    zoom: float = 12.0  # visible ground width at the image center (world units)
    tilt: float = 45.0  # 0 = straight down, higher = more 3/4 perspective
    stripe_count: int = 0  # 0 = stripes run all the way across the image
    stripe_width: float = 0.5
    stripe_gap: float = 0.5
    depth: float = 3.0  # length of each stripe across the walking lanes
    road_color: str = "#808080"
    stripe_color: str = "#ffffff"
    # crowd
    crowd_enabled: bool = True
    crowd_count: int = 12
    loop_pedestrians: bool = True
    person_size: float = 1.75  # world-unit height (stripe width is ~0.5)
    size_var: float = 0.12  # +/- fraction of person_size
    crowd_speed: float = 1.0
    crowd_seed: int = 1
    shadows_enabled: bool = True
    shadow_style: str = "contact"  # "contact" or "cast"
    anim_fps: int = 12
    anim_cycles: int = 4  # animation length in 12-frame walk cycles
    # dog (sits at the center of the image)
    dog_enabled: bool = True
    dog_size: float = 0.8  # world-unit height
    dog_clearance: float = 0.9  # radius people steer around
    dog_tilt: float = 12.0  # max lean in degrees when looking sideways
    dog_hold: int = 2  # output frames per dog drawing
    dog_blink: bool = True
    stripe_wobble: float = 0.35  # fraction of the allowed redraw displacement
    background_animated: bool = True
    stripes_animated: bool = True
    background_zoom: float = 1.0  # relative to the single-image cover scale
    background_perspective: bool = True
    background_images: tuple[str, ...] = ()  # empty uses the built-in texture
    custom_pedestrians: tuple[crowd.CustomPedestrian, ...] = ()
    camera_shift: float = 0.0  # camera-local up, as a percentage of zoom


def random_scene_params(params, rng=None):
    rng = rng or random
    p = replace(params)
    p.mode = rng.choice(("single", "cross"))
    p.angle = rng.uniform(0, 360)
    p.zoom = rng.uniform(5, 16)
    p.tilt = rng.uniform(20, 65)
    p.depth = rng.uniform(1.5, 8)
    p.road_color = rng.choice(
        ("#55585c", "#66625a", "#757575", "#48535b", "#827f75"))
    p.background_animated = rng.choice((True, False))
    p.background_zoom = rng.uniform(max(MIN_BACKGROUND_ZOOM, 0.5), 1.6)
    p.background_perspective = rng.choice((True, False))
    p.crowd_enabled = rng.choice((True, True, False))
    p.crowd_count = rng.randint(5, 24)
    p.loop_pedestrians = rng.choice((True, True, False))
    p.crowd_seed = rng.randrange(1_000_000)
    p.shadows_enabled = rng.choice((True, True, False))
    p.shadow_style = rng.choice(("contact", "cast"))
    if not p.crowd_enabled and not p.dog_enabled and not any(
            person.route for person in p.custom_pedestrians):
        p.crowd_enabled = True
    return p


def load_defaults():
    if not DEFAULTS_PATH.exists():
        return Params()

    values = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    return replace(decode_params(values), custom_pedestrians=())


def decode_pedestrians(raw):
    if not isinstance(raw, list):
        raise ValueError("Custom pedestrians must be a list.")
    pedestrians = []
    ids = set()
    keys = set(crowd.CustomPedestrian.__dataclass_fields__)
    for spec in raw:
        if not isinstance(spec, dict) or (set(spec) | {
                "height", "walk_duration", "pet_dog", "pet_seconds", "pet_offset",
                "side_walk_folder", "side_walk_stride"}) != keys:
            raise ValueError("Invalid custom pedestrian settings.")
        side_folder = spec.get("side_walk_folder")
        stride = spec.get("side_walk_stride", crowd.STRIDE_FACTOR)
        if (side_folder is not None and
                (not isinstance(side_folder, str) or not side_folder.strip() or "\0" in side_folder)):
            raise ValueError("Invalid imported walk-cycle folder.")
        if type(stride) not in (int, float) or not math.isfinite(stride) or stride <= 0:
            raise ValueError("Imported walk-cycle stride must be positive and finite.")
        pet_dog, pet_seconds = spec.get("pet_dog", False), spec.get("pet_seconds", 2.0)
        pet_offset = spec.get("pet_offset", (0.0, 0.0))
        if (type(pet_dog) is not bool or type(pet_seconds) not in (int, float)
                or not math.isfinite(pet_seconds) or pet_seconds <= 0
                or not isinstance(pet_offset, (list, tuple)) or len(pet_offset) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pet_offset)):
            raise ValueError("Invalid dog interaction duration or petting position.")
        height = spec.get("height")
        if height is not None and (type(height) not in (int, float)
                                   or not MIN_PERSON_SIZE <= height <= MAX_PERSON_SIZE):
            raise ValueError(f"Pedestrian size must be between {MIN_PERSON_SIZE:g} and {MAX_PERSON_SIZE:g}.")
        duration = spec.get("walk_duration")
        if duration is not None and (type(duration) is not int or duration < 2):
            raise ValueError("A Walk block must be at least two frames long.")
        pid, route, colors, female, speed, departure, clips = (
            spec[k] for k in ("id", "route", "colors", "female", "speed", "departure", "clips"))
        if (not isinstance(pid, str) or not pid or pid in ids
                or type(female) is not bool
                or type(speed) not in (int, float) or not 0.01 <= speed <= 100
                or type(departure) is not int or departure < 0
                or not isinstance(colors, (list, tuple)) or len(colors) != 4
                or not all(isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in colors)):
            raise ValueError("Invalid custom pedestrian identity, appearance, speed, or departure.")
        if (not isinstance(route, (list, tuple)) or (len(route) == 1 and side_folder is None)
                or any(not isinstance(point, (list, tuple)) or len(point) != 2
                       or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                       for point in route)):
            raise ValueError("A pedestrian route must be empty or contain at least two finite points; "
                             "an imported character may have one placement point.")
        route = tuple(tuple(point) for point in route)
        distances = crowd.route_distances(route)
        if not math.isfinite(distances[-1]) or any(b <= a for a, b in zip(distances, distances[1:])):
            raise ValueError("Consecutive route points must be distinct and have a finite distance.")
        blocks = []
        if not isinstance(clips, (list, tuple)):
            raise ValueError("Pedestrian clips must be a list.")
        for clip in clips:
            if not isinstance(clip, dict) or set(clip) != {"folder", "start", "end"}:
                raise ValueError("Invalid pedestrian animation block.")
            folder, start, end = clip["folder"], clip["start"], clip["end"]
            if (not isinstance(folder, str) or not folder.strip() or "\0" in folder
                    or type(start) is not int or type(end) is not int or not 0 <= start <= end):
                raise ValueError("Invalid pedestrian animation folder or frame range.")
            blocks.append(crowd.ClipBlock(folder, start, end))
        blocks.sort(key=lambda block: block.start)
        if any(a.end >= b.start for a, b in zip(blocks, blocks[1:])):
            raise ValueError("Animation blocks on one pedestrian cannot overlap.")
        if side_folder is not None and (blocks or pet_dog):
            raise ValueError("Imported pedestrians cannot use layered clips or dog petting.")
        ids.add(pid)
        pedestrians.append(crowd.CustomPedestrian(pid, route, tuple(colors), female, speed,
                                                  departure, tuple(blocks), height=height,
                                                  walk_duration=duration, pet_dog=pet_dog,
                                                  pet_seconds=pet_seconds, pet_offset=tuple(pet_offset),
                                                  side_walk_folder=side_folder, side_walk_stride=stride))
    return tuple(pedestrians)


def decode_params(values):
    if not isinstance(values, dict):
        raise ValueError("Saved defaults must be a JSON object.")

    defaults = asdict(Params())
    settings = {"shadow_style": "cast"}
    for key, value in values.items():
        if key not in defaults:
            raise ValueError(f"Unknown saved setting: {key}")
        default = defaults[key]
        if type(default) is bool:
            valid = type(value) is bool
        elif type(default) is int:
            valid = type(value) is int
        elif type(default) is float:
            valid = type(value) in (int, float) and math.isfinite(value)
            if valid:
                value = float(value)
        elif key == "custom_pedestrians":
            settings[key] = decode_pedestrians(value)
            continue
        elif isinstance(default, tuple):
            valid = isinstance(value, list) and all(
                isinstance(path, str) and path for path in value)
            if valid:
                value = tuple(value)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"Invalid value for saved setting: {key}")
        if key == "mode" and value not in ("single", "cross"):
            raise ValueError("Saved mode must be 'single' or 'cross'.")
        if key == "shadow_style" and value not in ("contact", "cast"):
            raise ValueError("Saved shadow style must be 'contact' or 'cast'.")
        if key in ("road_color", "stripe_color"):
            try:
                ImageColor.getrgb(value)
            except ValueError as e:
                raise ValueError(f"Invalid color for saved setting: {key}") from e
        settings[key] = value
    if "stripes_animated" not in settings and "background_animated" in settings:
        settings["stripes_animated"] = settings["background_animated"]
    return replace(Params(), **settings)


def save_defaults(params: Params):
    DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    values = asdict(params)
    del values["custom_pedestrians"]
    DEFAULTS_PATH.write_text(
        json.dumps(values, indent=2) + "\n", encoding="utf-8")


@dataclass
class Shot:
    params: Params
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Sequence:
    name: str
    shots: list[Shot] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Project:
    editor: Params
    sequences: list[Sequence]
    active_id: str


def new_project(params):
    sequence = Sequence("Sequence1")
    return Project(replace(params, custom_pedestrians=()), [sequence], sequence.id)


def validate_sequence_name(name):
    if (not isinstance(name, str) or not name.strip() or len(name) > 100
            or name != name.strip() or name.endswith(".")
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("Use a unique sequence name of 1–100 characters, without "
                         'Windows filename characters (<>:"/\\|?*), reserved device '
                         "names, or leading/trailing spaces or trailing periods.")
    return name


def validate_project_params(p):
    if p.shadow_style not in ("contact", "cast"):
        raise ValueError("Shadow style must be 'contact' or 'cast'.")
    for spec in p.custom_pedestrians:
        if spec.side_walk_folder is not None:
            if (not isinstance(spec.side_walk_folder, str) or not spec.side_walk_folder.strip()
                    or "\0" in spec.side_walk_folder):
                raise ValueError("Invalid imported walk-cycle folder.")
            if spec.clips or spec.pet_dog:
                raise ValueError("Imported pedestrians cannot use layered clips or dog petting.")
        if (type(spec.side_walk_stride) not in (int, float)
                or not math.isfinite(spec.side_walk_stride) or spec.side_walk_stride <= 0):
            raise ValueError("Imported walk-cycle stride must be positive and finite.")
    ranges = {
        "width": (16, None), "height": (16, None), "angle": (0, 360),
        "zoom": (1, 20), "tilt": (0, 70), "stripe_count": (0, 40),
        "camera_shift": (-75, 75),
        "stripe_width": (0.05, 3), "stripe_gap": (0.05, 3),
        "depth": (0.5, 15), "stripe_wobble": (0, 1),
        "background_zoom": (MIN_BACKGROUND_ZOOM, MAX_BACKGROUND_ZOOM),
        "crowd_count": (0, 100), "person_size": (MIN_PERSON_SIZE, MAX_PERSON_SIZE),
        "size_var": (0, 0.4), "crowd_speed": (0.25, 3),
        "crowd_seed": (0, None), "anim_fps": (1, None),
        "anim_cycles": (1, None), "dog_size": (0.2, 3),
        "dog_clearance": (0, 3), "dog_tilt": (0, 30), "dog_hold": (1, 6),
    }
    for key, (low, high) in ranges.items():
        value = getattr(p, key)
        if not math.isfinite(value) or value < low or (high is not None and value > high):
            raise ValueError(f"{key} must be at least {low}"
                             + (f" and at most {high}." if high is not None else "."))
    interaction_for(p)


def validate_backgrounds(params):
    for path in dict.fromkeys(path for p in params for path in p.background_images):
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as e:
            raise ValueError(f"Cannot read background image {path}:\n{e}") from e


def project_values(project, directory):
    def relative(path):
        absolute = Path(path).resolve()
        try:
            return os.path.relpath(absolute, directory)
        except ValueError:  # Paths on separate Windows drives cannot be relative.
            return str(absolute)

    def settings(p):
        validate_project_params(p)
        values = asdict(p)
        values["background_images"] = [relative(path) for path in p.background_images]
        for spec in values["custom_pedestrians"]:
            for block in spec["clips"]:
                block["folder"] = relative(block["folder"])
            if spec["side_walk_folder"] is not None:
                spec["side_walk_folder"] = relative(spec["side_walk_folder"])
        return values

    return {
        "version": 1,
        "editor": settings(project.editor),
        "active_sequence": project.active_id,
        "sequences": [
            {"id": sequence.id, "name": sequence.name,
             "shots": [{"id": shot.id, "params": settings(shot.params)}
                       for shot in sequence.shots]}
            for sequence in project.sequences],
    }


def save_project_file(project, path):
    path = Path(path).resolve()
    text = json.dumps(project_values(project, path.parent), indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_project_file(path):
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(values, dict) or type(values.get("version")) is not int
            or values["version"] != 1):
        raise ValueError("Unsupported or invalid CrossRoads project version.")

    def settings(raw):
        keys = set(asdict(Params()))
        if (not isinstance(raw, dict) or set(raw) - keys
                or keys - set(raw) - {"shadows_enabled", "shadow_style", "dog_blink",
                                      "background_animated", "stripes_animated",
                                      "custom_pedestrians", "camera_shift",
                                      "loop_pedestrians"}):
            raise ValueError("Project setups must contain every scene setting.")
        p = decode_params(raw)
        validate_project_params(p)
        p.background_images = tuple(
            str((path.parent / image).resolve()) for image in p.background_images)
        p.custom_pedestrians = tuple(
            replace(spec, clips=tuple(replace(block, folder=str((path.parent / block.folder).resolve()))
                                      for block in spec.clips),
                    side_walk_folder=(str((path.parent / spec.side_walk_folder).resolve())
                                      if spec.side_walk_folder is not None else None))
            for spec in p.custom_pedestrians)
        return p

    ids = set()

    def identity(raw):
        if not isinstance(raw, str) or not raw or raw in ids:
            raise ValueError("Project sequence and shot IDs must be unique strings.")
        ids.add(raw)
        return raw

    raw_sequences = values.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ValueError("A project must contain at least one sequence.")
    sequences = []
    names = set()
    for raw in raw_sequences:
        if not isinstance(raw, dict) or not isinstance(raw.get("shots"), list):
            raise ValueError("Invalid sequence or shot list.")
        name = validate_sequence_name(raw.get("name"))
        if name.casefold() in names:
            raise ValueError("Sequence names must be unique (ignoring case).")
        names.add(name.casefold())
        sequence = Sequence(name, id=identity(raw.get("id")))
        for item in raw["shots"]:
            if not isinstance(item, dict):
                raise ValueError("Invalid shot.")
            sequence.shots.append(Shot(settings(item.get("params")),
                                       identity(item.get("id"))))
        sequences.append(sequence)
    active = values.get("active_sequence")
    if active not in [sequence.id for sequence in sequences]:
        raise ValueError("The active sequence is missing from the project.")
    project = Project(settings(values.get("editor")), sequences, active)
    validate_backgrounds([project.editor] + [
        shot.params for sequence in sequences for shot in sequence.shots])
    return project


def make_dog(p: Params):
    if not p.dog_enabled:
        return None
    return dog_mod.Dog(height=p.dog_size, clearance=p.dog_clearance,
                       max_tilt=p.dog_tilt, hold=p.dog_hold,
                       blink_enabled=p.dog_blink, fps=p.anim_fps)


def interaction_for(p, *, validate=True):
    specs = [spec for spec in p.custom_pedestrians if spec.pet_dog]
    if not specs:
        return None
    if len(specs) > 1:
        raise ValueError("Only one pedestrian can interact with the dog in a shot.")
    spec = specs[0]
    if (type(spec.pet_seconds) not in (int, float) or not math.isfinite(spec.pet_seconds)
            or spec.pet_seconds <= 0 or not isinstance(spec.pet_offset, (tuple, list))
            or len(spec.pet_offset) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in spec.pet_offset)):
        raise ValueError("Invalid dog interaction duration or petting position.")
    if len(spec.route) < 2 or (validate and not p.dog_enabled):
        raise ValueError("Petting requires Show dog and a pedestrian route.")
    interaction = crowd.DogInteraction(spec, spec.height or p.person_size, p.anim_fps,
                                       Camera(p, p.width, p.height), p.dog_size)
    if validate and any(block.start < interaction.return_start and block.end >= interaction.pet_start
                        for block in spec.clips):
        raise ValueError("An animation clip overlaps the reserved petting interval.")
    return interaction


# ---------------------------------------------------------------- camera


class Camera:
    def __init__(self, p: Params, width: int, height: int):
        self.w, self.h = width, height
        t = math.radians(p.tilt)
        half = math.tan(math.radians(HFOV_DEG) / 2)
        self.dist = (p.zoom / 2) / half
        self.focal = (width / 2) / half
        self.fwd = (0.0, math.sin(t), -math.cos(t))
        self.up = (0.0, math.cos(t), math.sin(t))
        shift = p.zoom * p.camera_shift / 100
        self.pos = (0.0, -self.dist * self.fwd[1] + shift * self.up[1],
                    -self.dist * self.fwd[2] + shift * self.up[2])
        a = math.radians(p.angle)
        self.ca, self.sa = math.cos(a), math.sin(a)
        self.near = self.dist * 0.01

    def to_cam(self, x, y, z=0.0):
        # Rotate the scene by the crosswalk angle, then into camera space.
        rx = x * self.ca - y * self.sa
        ry = x * self.sa + y * self.ca
        qx, qy, qz = rx - self.pos[0], ry - self.pos[1], z - self.pos[2]
        cx = qx
        cy = qy * self.up[1] + qz * self.up[2]
        cz = qy * self.fwd[1] + qz * self.fwd[2]
        return cx, cy, cz

    def project(self, c):
        cx, cy, cz = c
        return (self.w / 2 + self.focal * cx / cz, self.h / 2 - self.focal * cy / cz)

    def ground_point(self, sx, sy):
        u = (sx - self.w / 2) / self.focal
        v = (self.h / 2 - sy) / self.focal
        dy, dz = self.fwd[1] + v * self.up[1], self.fwd[2] + v * self.up[2]
        if dz >= -1e-9:
            return None
        depth = -self.pos[2] / dz
        if depth <= self.near:
            return None
        rx, ry = depth * u, self.pos[1] + depth * dy
        return rx * self.ca + ry * self.sa, -rx * self.sa + ry * self.ca

    def ground_radius(self):
        """Radius around the origin that covers everything visible on the ground."""
        far = self.dist * 60
        best = 0.0
        t_fwd, t_up = self.fwd, self.up
        for sx, sy in ((0, 0), (self.w, 0), (0, self.h), (self.w, self.h)):
            u = (sx - self.w / 2) / self.focal
            v = -(sy - self.h / 2) / self.focal
            d = (u, t_fwd[1] + v * t_up[1], t_fwd[2] + v * t_up[2])
            if d[2] < -1e-9:
                s = -self.pos[2] / d[2]
                gx, gy = self.pos[0] + s * d[0], self.pos[1] + s * d[1]
                best = max(best, math.hypot(gx, gy))
            else:
                best = max(best, far)
        return min(best, far) + self.dist


def clip_near(poly, near):
    """Sutherland-Hodgman clip of a camera-space polygon against z >= near."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ain, bin_ = a[2] >= near, b[2] >= near
        if ain:
            out.append(a)
        if ain != bin_:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(tuple(a[k] + t * (b[k] - a[k]) for k in range(3)))
    return out


# -------------------------------------------------------------- geometry


def _bar_intervals(p: Params, radius: float):
    """Intervals [a, b] of stripes along a band axis, centered on 0."""
    pitch = p.stripe_width + p.stripe_gap
    if p.stripe_count > 0:
        n = p.stripe_count
        start = -(n * pitch - p.stripe_gap) / 2
        return [(start + i * pitch, start + i * pitch + p.stripe_width) for i in range(n)]
    k = min(int(radius / pitch) + 1, MAX_BARS_PER_BAND // 2)
    # Offset so a stripe is centered on the origin.
    return [(i * pitch - p.stripe_width / 2, i * pitch + p.stripe_width / 2) for i in range(-k, k + 1)]


def _subtract(a, b, lo, hi):
    pieces = []
    if a < lo:
        pieces.append((a, min(b, lo)))
    if b > hi:
        pieces.append((max(a, hi), b))
    return [(s, e) for s, e in pieces if e - s > 1e-9]


def _stripe_runs(p: Params, radius: float):
    """Yield (along_x, start, end) for each painted stripe segment."""
    h = p.depth / 2
    for a, b in _bar_intervals(p, radius):
        segs = _subtract(a, b, -h, h) if p.mode == "cross" else [(a, b)]
        for s, e in segs:
            yield True, s, e
        if p.mode == "cross":
            for s, e in segs:
                yield False, s, e


def _stripe_drawing(p: Params, frame: int) -> int:
    return frame // max(1, p.dog_hold) if p.stripe_wobble > 0 else 0


def _background_key(p: Params, frame: int) -> tuple[int, int, int]:
    drawing = _stripe_drawing(p, frame)
    stripe_drawing = drawing if p.stripes_animated else 0
    road_drawing = drawing if p.background_animated else 0
    image = ((frame // max(1, p.dog_hold)) % max(1, len(p.background_images))
             if p.background_animated else 0)
    return stripe_drawing, road_drawing, image


def _sketch_stripe(p: Params, along_x: bool, start: float, end: float,
                   unit: float, drawing: int):
    """Return a paint polygon and broken exterior strokes in band coordinates."""
    h = p.depth / 2
    key = f"{along_x}:{start.hex()}:{end.hex()}"
    rng = random.Random(key)
    motion = random.Random(f"{key}:{drawing}")
    wobble = min(1.0, max(0.0, p.stripe_wobble))
    dx, dy = (motion.uniform(-0.06, 0.06) * wobble * unit for _ in range(2))

    # Leave room for the strokes at the inner ends of the crossing arms.
    left = start + 0.38 * unit if p.mode == "cross" and start == h else start
    right = end - 0.38 * unit if p.mode == "cross" and end == -h else end
    corners = [
        (x + dx + unit * (rng.uniform(-0.07, 0.07) + wobble * motion.uniform(-0.04, 0.04)),
         y + dy + unit * (rng.uniform(-0.07, 0.07) + wobble * motion.uniform(-0.04, 0.04)))
        for x, y in ((left, -h), (right, -h), (right, h), (left, h))
    ]
    polygon, strokes = [], []
    for i, (nx, ny) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % 4]
        edge = [(ax, ay)]
        for j in range(1, 8):
            t = (j + rng.uniform(-0.25, 0.25)) / 8
            bend = unit * (rng.uniform(-0.065, 0.065) + wobble * motion.uniform(-0.05, 0.05))
            edge.append((ax + t * (bx - ax) + nx * bend, ay + t * (by - ay) + ny * bend))
        polygon.extend(edge)
        # A main stroke with an off-center break, plus a lighter retraced fragment.
        split = rng.choice((3, 4, 5))
        for points, offset, light in (
            (edge[:split], 0.040, False),
            (edge[split:], 0.040, False),
            (edge[split - 1:split + 2], 0.075, True),
        ):
            strokes.append(([
                (x + nx * unit * offset * rng.uniform(0.8, 1.2),
                 y + ny * unit * offset * rng.uniform(0.8, 1.2))
                for x, y in points
            ], light))
    return polygon, strokes


def walk_routes(p: Params) -> tuple[crowd.WalkRoute, ...]:
    """Keep both arms of each painted band on the same walking route."""
    bounds: dict[bool, tuple[float, float]] = {}
    for along_x, start, end in _stripe_runs(p, p.zoom / 2):
        lo, hi = bounds.get(along_x, (start, end))
        bounds[along_x] = min(lo, start), max(hi, end)
    return tuple(crowd.WalkRoute(along_x, lo, hi, p.depth / 2)
                 for along_x, (lo, hi) in bounds.items())


# -------------------------------------------------------------- renderer


@lru_cache(maxsize=4)
def _road_texture(road_color: str, image_path: str) -> Image.Image:
    with Image.open(image_path) as source:
        texture = source.convert("RGBA")
    # Transparent flecks reveal a dark undercoat.
    rgb = ImageColor.getcolor(road_color, "RGB")
    undercoat = (rgb[0] // 4, rgb[1] // 4, rgb[2] // 4)
    road = Image.new("RGB", texture.size, undercoat)
    road.paste(texture, mask=texture.getchannel("A"))
    return road


@lru_cache(maxsize=4)
def _tiled_road_texture(road_color: str, image_path: str) -> Image.Image:
    source = _road_texture(road_color, image_path).copy()
    source.thumbnail((1250, 1250), Image.Resampling.LANCZOS)
    pitch = max(1, min(source.size) * 4 // 5)
    overlap = pitch // 8
    size = pitch + overlap
    side = BACKGROUND_TILES * pitch
    tiled = Image.new("RGB", (side, side))

    # Fade only the edges that overlap already-painted patches.
    ramp = bytes(round(255 * i / max(1, overlap - 1)) for i in range(overlap))
    left = Image.frombytes("L", (size, 1), ramp + b"\xff" * pitch)
    left = left.resize((size, size), Image.Resampling.NEAREST)
    top = left.transpose(Image.Transpose.TRANSPOSE)
    masks = (None, left, top, ImageChops.multiply(left, top))

    rng = random.Random("road tiles")
    for row in range(BACKGROUND_TILES):
        for col in range(BACKGROUND_TILES):
            x = rng.randint(0, source.width - size)
            y = rng.randint(0, source.height - size)
            patch = source.crop((x, y, x + size, y + size))
            tiled.paste(patch, (col * pitch, row * pitch), masks[(col > 0) + 2 * (row > 0)])
    return tiled


def _draw_perspective_road(p: Params, texture: Image.Image, source_width: int,
                           width: int, height: int, cam: Camera,
                           drawing: int, zoom: float) -> Image.Image:
    source = np.asarray(texture, dtype=np.uint8)
    source_height, texture_width = source.shape[:2]
    output = np.empty((height, width, 3), dtype=np.uint8)
    screen_x = (np.arange(width, dtype=np.float64) + 0.5 - width / 2) / cam.focal
    scale = source_width / (p.zoom * zoom)
    wobble = min(1.0, max(0.0, p.stripe_wobble))
    angle = math.radians((drawing * BACKGROUND_TURN_DEG) % 360) if wobble else 0.0
    ca, sa = math.cos(angle), math.sin(angle)
    shift_limit = (0.12 * wobble * p.zoom * min(width, height) / width)
    rng = random.Random(f"road:{drawing}")
    dx, dy = (rng.uniform(-shift_limit, shift_limit) for _ in range(2))

    for row in range(0, height, 4):
        stop = min(row + 4, height)
        v = (cam.h / 2 - np.arange(row, stop)[:, None] - 0.5) / cam.focal
        denominator = -cam.fwd[2] - v * cam.up[2]
        ray_scale = np.full_like(denominator, cam.dist * 60)
        np.divide(cam.pos[2], denominator, out=ray_scale,
                  where=denominator > 1e-9)
        ray_scale = np.minimum(ray_scale, cam.dist * 60)
        scene_x = ray_scale * screen_x
        scene_y = cam.pos[1] + ray_scale * (v * cam.up[1] + cam.fwd[1])
        world_x = scene_x * cam.ca + scene_y * cam.sa
        world_y = -scene_x * cam.sa + scene_y * cam.ca
        texture_x = texture_width / 2 + scale * (
            ca * world_x + sa * world_y + dx)
        texture_y = source_height / 2 - scale * (
            -sa * world_x + ca * world_y + dy)

        x0_raw = np.floor(texture_x).astype(np.int32)
        y0_raw = np.floor(texture_y).astype(np.int32)
        fx = (texture_x - x0_raw)[..., None].astype(np.float32)
        fy = (texture_y - y0_raw)[..., None].astype(np.float32)
        x0, x1 = x0_raw % texture_width, (x0_raw + 1) % texture_width
        y0, y1 = y0_raw % source_height, (y0_raw + 1) % source_height
        top = source[y0, x0].astype(np.float32)
        top *= 1 - fx
        top += source[y0, x1].astype(np.float32) * fx
        bottom = source[y1, x0].astype(np.float32)
        bottom *= 1 - fx
        bottom += source[y1, x1].astype(np.float32) * fx
        top *= 1 - fy
        bottom *= fy
        top += bottom
        np.rint(top, out=top)
        np.clip(top, 0, 255, out=top)
        output[row:stop] = top.astype(np.uint8)
    return Image.fromarray(output)


def _draw_road(p: Params, width: int, height: int, drawing: int,
               image_path: str, cam: Camera) -> Image.Image:
    zoom = min(MAX_BACKGROUND_ZOOM, max(MIN_BACKGROUND_ZOOM, p.background_zoom))
    source = _road_texture(p.road_color, image_path)
    if zoom < 1.0:
        texture = _tiled_road_texture(p.road_color, image_path)
        tiles_across = BACKGROUND_TILES
    else:
        texture = source
        tiles_across = 1
    if p.background_perspective:
        return _draw_perspective_road(
            p, texture, source.width, width, height, cam, drawing, zoom)
    wobble = min(1.0, max(0.0, p.stripe_wobble))
    angle = math.radians((drawing * BACKGROUND_TURN_DEG) % 360) if wobble else 0.0
    shift_limit = 0.12 * wobble * min(width, height)
    rng = random.Random(f"road:{drawing}")
    dx, dy = (rng.uniform(-shift_limit, shift_limit) for _ in range(2))

    # The frame diagonal bounds the footprint at any rotation, so zoom won't pulse.
    if wobble:
        footprint_w = footprint_h = math.hypot(width, height)
    else:
        footprint_w, footprint_h = width, height
    scale = min(texture.width / (footprint_w + 2 * shift_limit),
                texture.height / (footprint_h + 2 * shift_limit)) / (tiles_across * zoom)
    c, s = scale * math.cos(angle), scale * math.sin(angle)
    cx, cy = width / 2, height / 2
    road = texture.transform(
        (width, height), Image.Transform.AFFINE,
        (c, s, texture.width / 2 + scale * dx - c * cx - s * cy,
         -s, c, texture.height / 2 + scale * dy + s * cx - c * cy),
        resample=Image.Resampling.BICUBIC,
    )
    return road


def _project_polygon(cam: Camera, poly, along_x: bool):
    return [cam.project(c) for c in clip_near(
        [cam.to_cam(x, y) if along_x else cam.to_cam(y, x) for x, y in poly], cam.near)]


def _draw_stroke(draw, cam: Camera, points, along_x: bool, color, width: int):
    a = cam.to_cam(*points[0] if along_x else points[0][::-1])
    for point in points[1:]:
        b = cam.to_cam(*point if along_x else point[::-1])
        if a[2] >= cam.near and b[2] >= cam.near:
            draw.line((cam.project(a), cam.project(b)), fill=color, width=width)
        elif (a[2] >= cam.near) != (b[2] >= cam.near):
            t = (cam.near - a[2]) / (b[2] - a[2])
            cut = tuple(x + t * (y - x) for x, y in zip(a, b))
            visible = a if a[2] >= cam.near else b
            draw.line((cam.project(visible), cam.project(cut)), fill=color, width=width)
        a = b


def render(p: Params, width=None, height=None, supersample=3, *, frame: int = 0) -> Image.Image:
    width = width or p.width
    height = height or p.height
    sw, shh = width * supersample, height * supersample
    cam = Camera(p, sw, shh)
    stripe_drawing, road_drawing, image_index = _background_key(p, frame)
    image_path = p.background_images[image_index] if p.background_images else str(ROAD_TEXTURE)
    img = _draw_road(p, sw, shh, road_drawing, image_path, cam)
    draw = ImageDraw.Draw(img)
    road = ImageColor.getcolor(p.road_color, "RGB")
    graphite = 28 if sum(road) >= 180 else 143
    pencil = tuple(tuple(round(c + (graphite - c) * strength) for c in road)
                   for strength in (0.65, 0.40))
    stroke_width = max(1, round(sw / 540))
    h = p.depth / 2
    for along_x, start, end in _stripe_runs(p, cam.ground_radius()):
        unit = min(end - start, p.depth, p.stripe_gap)
        pad = 0.5 * unit
        bounds = _project_polygon(cam, [
            (start - pad, -h - pad), (end + pad, -h - pad),
            (end + pad, h + pad), (start - pad, h + pad),
        ], along_x)
        if not bounds:
            continue
        xs, ys = zip(*bounds)
        if (max(xs) < -stroke_width or min(xs) > sw + stroke_width or
                max(ys) < -stroke_width or min(ys) > shh + stroke_width):
            continue
        polygon, strokes = _sketch_stripe(p, along_x, start, end, unit, stripe_drawing)
        for points, light in strokes:
            _draw_stroke(draw, cam, points, along_x, pencil[light],
                         max(1, stroke_width // 2) if light else stroke_width)
        pts = _project_polygon(cam, polygon, along_x)
        if len(pts) >= 3:
            draw.polygon(pts, fill=p.stripe_color)
    if supersample > 1:
        img = img.resize((width, height), Image.LANCZOS)
    return img


class ShotRenderer:
    def __init__(self, params, width=None, height=None, supersample=3, *, editor=False):
        self.p = replace(params)
        self.width = width or params.width
        self.height = height or params.height
        self.supersample = supersample
        self.cam = Camera(self.p, self.width, self.height)
        self.dog = make_dog(self.p)
        self.people = (crowd.make_crowd(self.p, walk_routes(self.p), self.cam,
                                       self.dog.obstacle() if self.dog else None)
                       if self.p.crowd_enabled and self.p.crowd_count > 0 else [])
        self.custom = [crowd.CustomWalker(spec, spec.height or self.p.person_size, self.p.anim_fps)
                       for spec in self.p.custom_pedestrians if spec.route]
        self.interaction = interaction_for(self.p)
        for walker in self.custom:
            if self.interaction is not None and walker.spec.pet_dog:
                walker.interaction = self.interaction
                walker.pet_hold = self.p.dog_hold
        self.editor = editor
        self.background = None
        self.background_key = None

    def frame(self, number):
        key = _background_key(self.p, number)
        if self.background is None or key != self.background_key:
            self.background = render(self.p, self.width, self.height,
                                     self.supersample, frame=number)
            self.background_key = key
        return self.compose(self.background, number)

    def compose(self, background, number):
        image = background.copy()
        crowd.draw_frame(image, self.cam, self.people, number, self.dog,
                         shadows=self.p.shadows_enabled, shadow_style=self.p.shadow_style,
                         shadow_hold=self.p.dog_hold, custom=self.custom, allow_missing=self.editor,
                         loop_pedestrians=self.p.loop_pedestrians)
        return image

    def __iter__(self):
        for number in range(self.p.anim_cycles * crowd.FRAME_COUNT):
            yield self.frame(number)


def sequence_outputs(sequence, directory):
    directory = Path(directory)
    prefix = f"{sequence.name}_shot".casefold()
    if any(path.name.casefold().startswith(prefix) and path.suffix.casefold() == ".png"
           for path in directory.iterdir()):
        raise ValueError(f"{directory} already contains PNG frames for {sequence.name}.\n"
                         "Choose another folder to avoid overwriting or mixing old frames.")
    validate_backgrounds([shot.params for shot in sequence.shots])
    for shot in sequence.shots:
        validate_project_params(shot.params)
        validate_animation_folders(shot.params)
    def frames():
        for index, shot in enumerate(sequence.shots, 1):
            for number, image in enumerate(ShotRenderer(shot.params), 1):
                yield directory / f"{sequence.name}_shot{index}.{number:04d}.png", image, index, number
    return frames()


def _thumbnail_image(p):
    scale = min(160 / p.width, 90 / p.height)
    return ShotRenderer(p, max(1, round(p.width * scale)),
                        max(1, round(p.height * scale)), supersample=2, editor=True).frame(0)


def _preview_images(p, width, height, frame):
    global _PREVIEW_RENDERER
    renderer = _PREVIEW_RENDERER
    if (renderer is None or renderer.p != p
            or renderer.width != width or renderer.height != height):
        renderer = _PREVIEW_RENDERER = ShotRenderer(p, width, height, supersample=2, editor=True)
    image = renderer.frame(frame)
    return renderer.background, image


def _pet_preview_images(p, width, height, frame):
    background, image = _preview_images(p, width, height, frame)
    cam = Camera(p, width, height)
    spec = next(spec for spec in p.custom_pedestrians if spec.pet_dog)
    person_height = spec.height or p.person_size
    _, cx, cy, pixel_height = crowd._walker_projection(
        cam, 0, 0, max(person_height, p.dog_size))
    scale = max(1.0, min(4.0, width / (2.5 * pixel_height), height / (1.7 * pixel_height)))
    crop_width, crop_height = width / scale, height / scale
    left = max(0.0, min(width - crop_width, cx - crop_width / 2))
    top = max(0.0, min(height - crop_height, cy - .4 * pixel_height - crop_height / 2))
    image = image.resize((width, height), Image.LANCZOS,
                         box=(left, top, left + crop_width, top + crop_height))
    return background, image, scale


def _clear_render_caches(cancelled, progress):
    global _PREVIEW_RENDERER
    _PREVIEW_RENDERER = None
    _road_texture.cache_clear()
    _tiled_road_texture.cache_clear()
    crowd.load_clip.cache_clear()
    crowd.load_side_walk.cache_clear()
    crowd.load_petting.cache_clear()
    dog_mod.load_interaction_assets.cache_clear()


def animation_folders(p):
    return dict.fromkeys(block.folder for spec in p.custom_pedestrians for block in spec.clips)


def side_walk_folders(p):
    return dict.fromkeys(spec.side_walk_folder for spec in p.custom_pedestrians
                         if spec.side_walk_folder is not None)


def validate_animation_folders(p):
    if interaction_for(p) is not None:
        crowd.load_petting()
        dog_mod.load_interaction_assets()
    for folder in animation_folders(p):
        if not Path(folder).is_dir():
            raise ValueError(f"Animation folder is missing. Relink it in Custom Pedestrians:\n{folder}")
        crowd.load_clip(folder)
    for folder in side_walk_folders(p):
        if not Path(folder).is_dir():
            raise ValueError(f"Walk-cycle folder is missing. Relink it in Custom Pedestrians:\n{folder}")
        crowd.load_side_walk(folder)


def _background_call(function, args, cancelled, progress):
    if cancelled.is_set():
        return None
    return function(*args)


def _export_sequence(sequence, directory, cancelled, progress):
    if cancelled.is_set():
        return "Cancelled"
    outputs = sequence_outputs(sequence, directory)
    try:
        written = 0
        while not cancelled.is_set():
            output = next(outputs, None)
            if output is None:
                return "Complete"
            path, image, shot, frame = output
            if cancelled.is_set():
                break
            with path.open("xb") as handle:
                image.save(handle, format="PNG")
            written += 1
            progress((written, f"Rendering shot {shot}, frame {frame}"))
        return "Cancelled"
    finally:
        outputs.close()


def _export_animation(p, base, cancelled, progress):
    if cancelled.is_set():
        return "Cancelled"
    validate_animation_folders(p)
    renderer = ShotRenderer(p)
    gif_scale = min(1.0, 640 / p.width)
    gif_size = (max(1, int(p.width * gif_scale)), max(1, int(p.height * gif_scale)))
    gif_frames = []
    for t in range(p.anim_cycles * crowd.FRAME_COUNT):
        if cancelled.is_set():
            return "Cancelled"
        image = renderer.frame(t)
        if cancelled.is_set():
            return "Cancelled"
        image.save(f"{base}_{t + 1:04d}.png")
        progress((t + 1, f"Rendering frame {t + 1}"))
        small = image.resize(gif_size, Image.LANCZOS) if gif_scale < 1.0 else image
        gif_frames.append(small.convert("P", palette=Image.ADAPTIVE, colors=256))
    if cancelled.is_set():
        return "Cancelled"
    progress((len(gif_frames), "Encoding GIF (cancellation waits for encoding)"))
    loop = {} if renderer.people or renderer.custom else {"loop": 0}
    gif_frames[0].save(
        base + ".gif", save_all=True, append_images=gif_frames[1:],
        duration=int(round(1000 / p.anim_fps)), **loop)
    return "Complete"


# -------------------------------------------------------------------- UI


class ClipDialog(simpledialog.Dialog):
    def __init__(self, parent, folders, block, last_frame, other_blocks):
        self.folders = folders
        self.block = block
        self.last_frame = last_frame
        self.other_blocks = other_blocks
        super().__init__(parent, "Pedestrian animation block")

    def body(self, master):
        self.folder = tk.StringVar(value=self.block.folder)
        self.start = tk.StringVar(value=str(self.block.start + 1))
        self.end = tk.StringVar(value=str(self.block.end + 1))
        ttk.Label(master, text="Layered PNG folder").grid(row=0, column=0, sticky="w")
        choice = ttk.Combobox(master, textvariable=self.folder, values=self.folders, width=48)
        choice.grid(row=1, column=0, sticky="ew")
        ttk.Button(master, text="Browse...", command=self._browse).grid(row=1, column=1)
        for row, label, var in ((2, "First frame", self.start), (3, "Last frame (inclusive)", self.end)):
            ttk.Label(master, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(master, textvariable=var, width=8).grid(row=row, column=1)
        return choice

    def _browse(self):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            self.folder.set(folder)

    def validate(self):
        try:
            try:
                start, end = int(self.start.get()) - 1, int(self.end.get()) - 1
            except ValueError as error:
                raise ValueError("Enter whole frame numbers.") from error
            if not 0 <= start <= end < self.last_frame:
                raise ValueError(f"Use an inclusive frame range from 1 to {self.last_frame}.")
            if any(start <= block.end and block.start <= end for block in self.other_blocks):
                raise ValueError("Animation blocks on one pedestrian cannot overlap.")
            if not self.folder.get().strip():
                raise ValueError("Choose a layered PNG folder.")
            folder = str(Path(self.folder.get()).resolve())
            crowd.load_clip(folder)
        except (OSError, ValueError) as error:
            messagebox.showerror("Invalid animation block", str(error), parent=self)
            return False
        self.result = crowd.ClipBlock(folder, start, end)
        return True


class App:
    TIMELINE_LEFT = 63
    TIMELINE_LANES = {"walk": (24, 45), "clip": (50, 71)}
    PREVIEW_MAX = (900, 650)

    def __init__(self, root):
        self.root = root
        root.title("Crosswalk Generator")
        try:
            self.p = load_defaults()
        except (OSError, ValueError, TypeError, OverflowError) as e:
            self.p = Params()
            messagebox.showerror(
                "Could not load defaults",
                f"Using built-in defaults. Could not read {DEFAULTS_PATH}:\n{e}",
                parent=root,
            )
        self._job = None
        self._photo = None
        self._preview_bg = None
        self._preview_cam = None
        self._preview_bounds = None
        self._preview_scale = 1.0
        self._preview_background_key = None
        self._preview_frame = 0
        self._anim_t = 0
        self._anim_job = None
        self._scrub_frame = 0
        self._block_drag = None
        self._timeline_scrubbing = False
        self._timeline_notice = ""
        self._pedestrian_id = None
        self._route_draft = None
        self._positioning_pet = False
        self._pet_drag = None
        self.project = new_project(self.p)
        self.project_path = None
        self._saved_state = None
        self._edit_id = None
        self._thumbnails = {}
        self._loading = True
        self._rendering = False
        self._drag_id = None
        self._drag_target = None
        self._disabled_widgets = []
        self._executor = _BackgroundWorker()
        self._future = None
        self._work_revision = 0
        self._worker_job = None
        self._worker_cancel = self._executor.cancelled
        self._progress = self._executor.progress
        self._pending_operation = None
        self._pending_preview = None
        self._pending_thumbnails = {}
        self._pending_cache_clear = False
        self._project_revision = 0
        self._preview_revision = 0
        self._thumbnail_labels = {}
        self._cards_job = None
        self._cards_revision = 0
        self._closed = False
        root.bind("<Destroy>", self._on_destroy, add="+")

        ctrl = ttk.Frame(root, padding=10)
        ctrl.grid(row=0, column=0, rowspan=2, sticky="ns")
        self.canvas = tk.Canvas(root, width=self.PREVIEW_MAX[0], height=430, bg="#222", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda e: self.schedule())
        self.canvas.bind("<Button-1>", self._place_route_point)
        self.canvas.bind("<B1-Motion>", self._drag_pet_position)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_pet_drag", None))

        self.vars = {}

        nb = self.notebook = ttk.Notebook(ctrl)
        nb.grid(row=1, column=0, sticky="nsew")
        walk_tab = ttk.Frame(nb, padding=8)
        crowd_tab = ttk.Frame(nb, padding=8)
        dog_tab = ttk.Frame(nb, padding=8)
        self.pedestrian_tab = ttk.Frame(nb, padding=8)
        nb.add(walk_tab, text="Crosswalk")
        nb.add(crowd_tab, text="Crowd")
        nb.add(dog_tab, text="Dog")
        nb.add(self.pedestrian_tab, text="Custom Pedestrians")
        r = 0

        ttk.Label(walk_tab, text="Image size (px)", font=("", 9, "bold")).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        size = ttk.Frame(walk_tab)
        size.grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 8)); r += 1
        for i, (key, label) in enumerate((("width", "W"), ("height", "H"))):
            v = tk.IntVar(value=getattr(self.p, key))
            self.vars[key] = v
            ttk.Label(size, text=label).grid(row=0, column=i * 2, padx=(0 if i == 0 else 8, 2))
            ttk.Spinbox(size, from_=16, to=8192, increment=16, width=7, textvariable=v, command=self.schedule).grid(row=0, column=i * 2 + 1)
            v.trace_add("write", lambda *a: self.schedule())

        ttk.Label(walk_tab, text="Mode", font=("", 9, "bold")).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        mv = tk.StringVar(value=self.p.mode)
        self.vars["mode"] = mv
        mf = ttk.Frame(walk_tab)
        mf.grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 8)); r += 1
        ttk.Radiobutton(mf, text="Single crosswalk", value="single", variable=mv, command=self.schedule).pack(side="left")
        ttk.Radiobutton(mf, text="Cross (+)", value="cross", variable=mv, command=self.schedule).pack(side="left", padx=8)

        sliders = [
            ("angle", "Angle (°)", 0, 360, float),
            ("zoom", "Zoom out (visible width)", 1, 20, float),
            ("tilt", "Camera tilt (°)", 0, 70, float),
            ("camera_shift", "Camera up/down (%)", -75, 75, float),
            ("stripe_count", "Stripe count (0 = full)", 0, 40, int),
            ("stripe_width", "Stripe width", 0.05, 3, float),
            ("stripe_gap", "Stripe gap", 0.05, 3, float),
            ("depth", "Crosswalk depth", 0.5, 15, float),
            ("stripe_wobble", "Stripe / texture wobble (0 = still)", 0, 1, float),
            ("background_zoom", "Background zoom (tiles below 1.00)", MIN_BACKGROUND_ZOOM, MAX_BACKGROUND_ZOOM, float),
        ]
        for key, label, lo, hi, typ in sliders:
            r = self._slider(walk_tab, r, key, label, lo, hi, typ)

        background_animated = tk.BooleanVar(value=self.p.background_animated)
        self.vars["background_animated"] = background_animated
        ttk.Checkbutton(
            walk_tab, text="Animate background",
            variable=background_animated, command=self.schedule,
        ).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1

        stripes_animated = tk.BooleanVar(value=self.p.stripes_animated)
        self.vars["stripes_animated"] = stripes_animated
        ttk.Checkbutton(
            walk_tab, text="Animate stripes",
            variable=stripes_animated, command=self.schedule,
        ).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1

        background_perspective = tk.BooleanVar(value=self.p.background_perspective)
        self.vars["background_perspective"] = background_perspective
        ttk.Checkbutton(
            walk_tab, text="Perspective background (match camera)",
            variable=background_perspective, command=self.schedule,
        ).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1

        background_controls = ttk.Frame(walk_tab)
        background_controls.grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 0)); r += 1
        ttk.Button(background_controls, text="Choose backgrounds…", command=self.choose_backgrounds).pack(side="left")
        ttk.Button(background_controls, text="Use default", command=self.reset_background).pack(side="left", padx=(4, 0))
        self.background_label = ttk.Label(walk_tab, wraplength=250)
        self.background_label.grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        self._update_background_label()

        ttk.Label(walk_tab, text="Colors", font=("", 9, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 0)); r += 1
        for key, label in (("road_color", "Road base"), ("stripe_color", "Stripes")):
            btn = tk.Button(walk_tab, text=label, width=10, bg=getattr(self.p, key), command=lambda k=key: self.pick_color(k))
            btn.grid(row=r, column=0, columnspan=3, sticky="w", pady=2); r += 1
            setattr(self, f"{key}_btn", btn)
            self._fix_btn_fg(btn, getattr(self.p, key))

        self.vars["animate_preview"] = tk.BooleanVar(value=False)
        self._animate_checkbox(walk_tab, r)
        self._build_crowd_tab(crowd_tab)
        self._build_dog_tab(dog_tab)
        self._build_pedestrian_tab(self.pedestrian_tab)
        nb.bind("<<NotebookTabChanged>>", self._pedestrian_tab_changed)

        scene_actions = ttk.Frame(ctrl)
        scene_actions.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(scene_actions, text="Set defaults", command=self.set_defaults).pack(
            side="left", fill="x", expand=True)
        ttk.Button(scene_actions, text="Generate random scene",
                   command=self.generate_random_scene).pack(
                       side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(ctrl, text="Save PNG…", command=self.save).grid(row=2, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(ctrl, text="Save Animation…", command=self.save_animation).grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self.status = ttk.Label(ctrl, text="", foreground="#666", wraplength=260)
        self.status.grid(row=4, column=0, sticky="w", pady=(6, 0))

        self._build_sequence_panel()
        self._build_project_menu()
        for variable in self.vars.values():
            variable.trace_add("write", self._editor_changed)
        self._loading = False
        self._saved_state = self._project_state()
        root.protocol("WM_DELETE_WINDOW", self.close_project)
        self.schedule()

    def _on_destroy(self, event):
        if event.widget != self.root:
            return
        self._closed = True
        if self._worker_cancel is not None:
            self._worker_cancel.set()
        for job in (self._worker_job, self._job, self._anim_job, self._cards_job):
            if job is not None:
                self.root.after_cancel(job)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _ensure_worker_poll(self):
        if not self._closed and self._worker_job is None:
            self._worker_job = self.root.after(15, self._poll_worker)

    def _poll_worker(self):
        self._worker_job = None
        self._drain_progress()
        if self._future is not None and self._future.done():
            self._drain_progress()
            future, kind, context = self._future, self._work_kind, self._work_context
            cancelled = self._worker_cancel.is_set()
            self._future = None
            try:
                response = future.result()
            except Exception as error:
                # This is the worker boundary: every failure is shown on the Tk thread.
                if not cancelled or kind in ("load", "sequence", "animation"):
                    self._work_failed(kind, context, error)
            else:
                self._drain_progress()
                if response.progress is not None:
                    self._show_progress(response.progress)
                if response.error is not None:
                    if not cancelled or kind in ("load", "sequence", "animation"):
                        self._work_failed(kind, context, response.error)
                else:
                    self._work_finished(kind, context, response.value, cancelled)
        if self._future is None:
            request = None
            if self._pending_cache_clear:
                request = ("clear", _clear_render_caches, None)
                self._pending_cache_clear = False
            elif self._pending_operation is not None:
                request = self._pending_operation
                self._pending_operation = None
            elif not self._rendering and self._pending_preview is not None:
                request = self._pending_preview
                self._pending_preview = None
            elif not self._rendering and self._pending_thumbnails:
                left = self.sequence_canvas.canvasx(0)
                right = left + self.sequence_canvas.winfo_width()
                shot_id = next((
                    sid for sid in self._pending_thumbnails
                    if sid in self._thumbnail_labels
                    and self._thumbnail_labels[sid].master.winfo_x() <= right
                    and self._thumbnail_labels[sid].master.winfo_x()
                    + self._thumbnail_labels[sid].master.winfo_width() >= left
                ), next(iter(self._pending_thumbnails)))
                request = self._pending_thumbnails.pop(shot_id)
            if request is not None:
                self._work_kind, function, self._work_context = request
                self._worker_cancel.clear()
                if self._rendering and self._operation_cancel.is_set():
                    self._worker_cancel.set()
                self._work_revision += 1
                try:
                    self._future = self._executor.submit(function, self._work_revision)
                except Exception as error:
                    self._work_failed(self._work_kind, self._work_context, error)
        if (self._future is not None or self._pending_operation or self._pending_cache_clear
                or self._pending_preview or self._pending_thumbnails):
            self._ensure_worker_poll()

    def _drain_progress(self):
        while True:
            try:
                token, progress = self._progress.get_nowait()
            except queue.Empty:
                break
            if self._future is not None and token == self._work_revision:
                self._show_progress(progress)

    def _show_progress(self, progress):
        written, message = progress
        self._render_written = written
        self.status.config(text=f"{message}\n{written}/{self._render_total} frames saved")

    def _work_finished(self, kind, context, result, cancelled):
        if kind == "load":
            self._finish_operation("Cancelled" if cancelled else "Complete")
            if not cancelled:
                if self._allow_replace_project():
                    self._install_project(result, context)
                elif self._resume_preview:
                    self.vars["animate_preview"].set(True)
                    self._toggle_anim_preview()
        elif kind in ("sequence", "animation"):
            self._finish_operation(result)
        elif kind == "thumbnail":
            revision, shot_id, params = context
            shot = next((s for seq in self.project.sequences for s in seq.shots
                         if s.id == shot_id), None)
            if (not cancelled and revision == self._project_revision
                    and shot is not None and shot.params == params):
                photo = ImageTk.PhotoImage(result, master=self.root)
                self._thumbnails[shot_id] = photo
                self._pending_thumbnails.pop(shot_id, None)
                label = self._thumbnail_labels.get(shot_id)
                if label is not None and label.winfo_exists():
                    label.config(image=photo, text="")
        elif kind == "preview":
            revision, params, width, height, frame, cw, ch = context
            if not cancelled and revision == self._preview_revision and not self._rendering:
                bg, image = result[:2]
                self._preview_scale = result[2] if len(result) == 3 else 1.0
                self._preview_bg = bg
                self._preview_background_key = _background_key(params, frame)
                self._preview_cam = Camera(params, width, height)
                self._photo = ImageTk.PhotoImage(image, master=self.root)
                self.canvas.delete("all")
                item = self.canvas.create_image(cw // 2, ch // 2, image=self._photo)
                self._preview_bounds = self.canvas.bbox(item)
                self._preview_frame = frame
                missing = any(not Path(folder).is_dir() for folder in
                              (*animation_folders(params), *side_walk_folders(params)))
                warning = "\nMissing animation folder: relink it in Custom Pedestrians." if missing else ""
                zoom = f"  (pet positioning {self._preview_scale:.1f}x)" if self._positioning_pet else ""
                self.status.config(text=f"Preview {width}x{height}  (output {params.width}x{params.height}){zoom}{warning}")
                self._draw_route_overlay()

    def _work_failed(self, kind, context, error):
        if kind in ("load", "sequence", "animation"):
            self._finish_operation("Failed")
            title = "Could not open project" if kind == "load" else "Export failed"
            details = str(error)
            if kind != "load":
                details += (f"\n\n{self._render_written} complete frames retained in "
                            f"{self._render_directory}. A failed frame may be incomplete.")
            messagebox.showerror(title, details, parent=self.root)
        elif kind == "thumbnail":
            revision, shot_id, params = context
            shot = next((s for seq in self.project.sequences for s in seq.shots
                         if s.id == shot_id), None)
            if revision == self._project_revision and shot is not None and shot.params == params:
                label = self._thumbnail_labels.get(shot_id)
                if label is not None and label.winfo_exists():
                    label.config(text="Preview unavailable")
                messagebox.showerror("Could not render shot preview", str(error), parent=self.root)
        elif kind == "preview" and context[0] == self._preview_revision:
            self._stop_preview_on_error(error)
        elif kind == "clear":
            messagebox.showerror("Could not clear render caches", str(error), parent=self.root)

    def _begin_operation(self, kind, function, context=None):
        self._resume_preview = self.vars["animate_preview"].get()
        self.vars["animate_preview"].set(False)
        self._toggle_anim_preview_stop()
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None
        self._preview_revision += 1
        self._pending_preview = None
        if self._worker_cancel is not None:
            self._worker_cancel.set()
        self._operation_cancel = threading.Event()
        self._operation_kind = kind
        self._rendering = True
        self._pending_operation = (kind, function, context)
        self._set_render_controls(True)
        self.status.config(text="Opening project..." if kind == "load" else "Preparing export...")
        self._ensure_worker_poll()

    def _finish_operation(self, outcome):
        kind = self._operation_kind
        self._rendering = False
        self._set_render_controls(False)
        if kind == "load":
            self.status.config(text=f"Project loading: {outcome.lower()}")
        else:
            text = (f"{outcome}: {self._render_written}/{self._render_total} PNG frames\n"
                    f"Output: {self._render_directory}")
            if kind == "animation" and outcome == "Complete":
                text += f"\nGIF: {self._animation_base}.gif"
            self.status.config(text=text)
        if self._resume_preview and (kind != "load" or outcome != "Complete"):
            self.vars["animate_preview"].set(True)
            self.update_preview()
            self._toggle_anim_preview()

    def set_defaults(self):
        try:
            save_defaults(replace(self.read_params()))
        except (OSError, TypeError, ValueError, tk.TclError) as e:
            messagebox.showerror("Could not save defaults", str(e), parent=self.root)
            return
        self.status.config(text="Current values saved as defaults.")

    def generate_random_scene(self):
        self._load_editor(random_scene_params(self.read_params()))
        self.status.config(text="Generated a random scene.")

    @property
    def sequence(self):
        return next(sequence for sequence in self.project.sequences
                    if sequence.id == self.project.active_id)

    def _build_project_menu(self):
        menu = tk.Menu(self.root)
        self.project_menu = tk.Menu(menu, tearoff=False)
        for label, command in (
                ("New project", self.new_project), ("Open project...", self.open_project),
                ("Save project", self.save_project), ("Save project as...", self.save_project_as)):
            self.project_menu.add_command(label=label, command=command)
        menu.add_cascade(label="Project", menu=self.project_menu)
        self.root.config(menu=menu)

    def _build_sequence_panel(self):
        panel = ttk.LabelFrame(self.root, text="Sequence", padding=6)
        panel.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 10))
        panel.columnconfigure(0, weight=1)
        toolbar = ttk.Frame(panel)
        toolbar.grid(row=0, column=0, sticky="ew")
        self.sequence_choice = ttk.Combobox(toolbar, state="readonly", width=18)
        self.sequence_choice.pack(side="left")
        self.sequence_choice.bind("<<ComboboxSelected>>", self._select_sequence)
        for label, command in (
                ("New", self.add_sequence), ("Rename", self.rename_sequence),
                ("Delete", self.delete_sequence), ("Add to sequence", self.add_shot),
                ("Render sequence...", self.render_sequence)):
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=(4, 0))
        self.cancel_render_button = ttk.Button(toolbar, text="Cancel task",
                                               command=self.cancel_sequence_render,
                                               state="disabled")
        self.cancel_render_button.pack(side="left", padx=(4, 0))
        editing = ttk.Frame(panel)
        editing.grid(row=1, column=0, sticky="ew", pady=4)
        self.edit_label = ttk.Label(editing, text="Editor changes do not alter saved shots.",
                                    wraplength=500)
        self.edit_label.pack(side="left")
        self.update_shot_button = ttk.Button(editing, text="Update shot",
                                              command=self.update_shot, state="disabled")
        self.update_shot_button.pack(side="left", padx=6)
        self.done_shot_button = ttk.Button(editing, text="Done editing",
                                            command=self.done_editing, state="disabled")
        self.done_shot_button.pack(side="left")
        self.sequence_canvas = tk.Canvas(panel, height=188, highlightthickness=0)
        self.sequence_canvas.grid(row=2, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(panel, orient="horizontal",
                                  command=self.sequence_canvas.xview)
        scrollbar.grid(row=3, column=0, sticky="ew")
        self.sequence_canvas.configure(xscrollcommand=scrollbar.set)
        self.cards = ttk.Frame(self.sequence_canvas)
        self.sequence_canvas.create_window(0, 0, window=self.cards, anchor="nw")
        self.cards.bind("<Configure>", lambda event: self.sequence_canvas.configure(
            scrollregion=self.sequence_canvas.bbox("all")))
        self._refresh_sequences()

    def _current_params(self):
        for key, variable in self.vars.items():
            if key != "animate_preview":
                value = variable.get()
                minimum = {"width": 16, "height": 16, "anim_fps": 1,
                           "anim_cycles": 1, "crowd_seed": 0}.get(key)
                if minimum is not None and value < minimum:
                    raise ValueError(f"{key} must be at least {minimum}.")
        p = replace(self.read_params())
        validate_project_params(p)
        return p

    def _project_state(self):
        self.project.editor = self._current_params()
        return json.dumps(project_values(self.project, Path.cwd()), sort_keys=True)

    def _editor_changed(self, *_):
        if self._loading:
            return
        try:
            dirty = self._project_state() != self._saved_state
        except (ValueError, TypeError, tk.TclError):
            dirty = True
        name = self.project_path.name if self.project_path else "Untitled project"
        self.root.title(f"{'* ' if dirty else ''}{name} - CrossRoads")
        self._update_edit_label()

    def _editing_shot(self):
        return next((shot for sequence in self.project.sequences for shot in sequence.shots
                     if shot.id == self._edit_id), None)

    def _update_edit_label(self):
        shot = self._editing_shot()
        if shot is None:
            text = "Editor changes do not alter saved shots."
        else:
            sequence = next(s for s in self.project.sequences if shot in s.shots)
            text = (f"Editing {sequence.name}, shot {sequence.shots.index(shot) + 1}. "
                    "Update shot to apply changes (project save keeps editor separate).")
        self.edit_label.config(text=text)
        state = "normal" if shot is not None and not self._rendering else "disabled"
        self.update_shot_button.config(state=state)
        self.done_shot_button.config(state=state)

    def _refresh_sequences(self):
        self.sequence_choice.config(values=[s.name for s in self.project.sequences])
        self.sequence_choice.current(self.project.sequences.index(self.sequence))
        self._refresh_cards()

    def _refresh_cards(self):
        self._cards_revision += 1
        if self._cards_job is not None:
            self.root.after_cancel(self._cards_job)
            self._cards_job = None
        self._drag_id = None
        self._drag_target = None
        self._pending_thumbnails.clear()
        self._thumbnail_labels.clear()
        for widget in self.cards.winfo_children():
            widget.destroy()
        self._card_widgets = []
        if not self.sequence.shots:
            ttk.Label(self.cards, text="Set up a shot, then choose Add to sequence.",
                      padding=25).pack()
        self._build_cards(list(self.sequence.shots), 0, self._cards_revision)
        self._update_edit_label()

    def _build_cards(self, shots, start, revision):
        self._cards_job = None
        if revision != self._cards_revision:
            return
        stop = min(start + 8, len(shots))
        for index in range(start, stop):
            shot = shots[index]
            card = tk.Frame(self.cards, bd=2, relief="solid",
                            highlightthickness=2, highlightbackground="#999")
            card.pack(side="left", padx=4, pady=2)
            self._card_widgets.append(card)
            photo = self._thumbnails.get(shot.id)
            label = ttk.Label(card, image=photo,
                              text="" if photo is not None else "Loading preview...")
            label.pack()
            self._thumbnail_labels[shot.id] = label
            if photo is None:
                params = replace(shot.params)
                self._pending_thumbnails[shot.id] = (
                    "thumbnail", partial(_background_call, _thumbnail_image, (params,)),
                    (self._project_revision, shot.id, params))
            p = shot.params
            details = ttk.Label(card, text=f"Shot {index + 1}  |  {p.width} x {p.height}\n"
                                f"{p.anim_cycles * crowd.FRAME_COUNT} frames  |  {p.anim_fps} FPS",
                                anchor="center")
            details.pack(fill="x")
            controls = ttk.Frame(card)
            controls.pack()
            for text, command in (
                    ("<", lambda sid=shot.id: self.move_shot(sid, -1)),
                    (">", lambda sid=shot.id: self.move_shot(sid, 1)),
                    ("Edit", lambda sid=shot.id: self.edit_shot(sid)),
                    ("Remove", lambda sid=shot.id: self.remove_shot(sid))):
                button = ttk.Button(controls, text=text, width=6, command=command)
                button.pack(side="left")
                if self._rendering:
                    self._disabled_widgets.append((button, "normal"))
                    button.config(state="disabled")
            for widget in (card, label, details):
                widget.bind("<ButtonPress-1>", lambda event, sid=shot.id: self._start_drag(sid))
                widget.bind("<B1-Motion>", self._drag_motion)
                widget.bind("<ButtonRelease-1>", self._drop_shot)
        if stop < len(shots):
            self._cards_job = self.root.after(
                1, self._build_cards, shots, stop, revision)
        self._ensure_worker_poll()

    def _start_drag(self, shot_id):
        if not self._rendering:
            self._drag_id = shot_id

    def _drag_motion(self, event):
        if self._drag_id is None or self._rendering:
            return
        x = event.x_root
        left = self.sequence_canvas.winfo_rootx()
        right = left + self.sequence_canvas.winfo_width()
        if x < left + 24:
            self.sequence_canvas.xview_scroll(-1, "units")
        elif x > right - 24:
            self.sequence_canvas.xview_scroll(1, "units")
        target = len(self._card_widgets)
        for index, widget in enumerate(self._card_widgets):
            if x < widget.winfo_rootx() + widget.winfo_width() / 2:
                target = index
                break
        self._drag_target = target
        for index, widget in enumerate(self._card_widgets):
            marked = index == min(target, len(self._card_widgets) - 1)
            widget.config(highlightbackground="#267bd9" if marked else "#999")

    def _drop_shot(self, event):
        if self._drag_id is not None and self._drag_target is not None and not self._rendering:
            shots = self.sequence.shots
            old = next(i for i, shot in enumerate(shots) if shot.id == self._drag_id)
            target = self._drag_target - (old < self._drag_target)
            shots.insert(target, shots.pop(old))
            self._refresh_cards()
            self._editor_changed()
        self._drag_id = None
        self._drag_target = None

    def move_shot(self, shot_id, offset):
        if self._rendering:
            return
        shots = self.sequence.shots
        index = next(i for i, shot in enumerate(shots) if shot.id == shot_id)
        target = index + offset
        if 0 <= target < len(shots):
            shots.insert(target, shots.pop(index))
            self._refresh_cards()
            self._editor_changed()

    def _allow_leave_shot(self):
        shot = self._editing_shot()
        try:
            changed = shot is not None and self._current_params() != shot.params
        except (ValueError, TypeError, tk.TclError) as e:
            messagebox.showerror("Invalid editor setup", str(e), parent=self.root)
            return False
        if changed:
            choice = messagebox.askyesnocancel(
                "Uncommitted shot edits", "Update the saved shot before leaving?\n"
                "Yes: update shot. No: leave the saved shot unchanged. Cancel: keep editing.",
                parent=self.root)
            if choice is None:
                return False
            if choice and not self.update_shot():
                return False
        return True

    def _load_editor(self, p):
        self._loading = True
        try:
            self.vars["animate_preview"].set(False)
            self._toggle_anim_preview_stop()
            self.p = replace(p)
            self._preview_bg = None
            self._preview_cam = self._preview_bounds = None
            self._preview_scale = 1.0
            self._preview_frame = 0
            self._scrub_frame = 0
            self._block_drag = None
            self._timeline_scrubbing = False
            self._timeline_notice = ""
            self._pedestrian_id = None
            self._route_draft = None
            self._positioning_pet = False
            self._pet_drag = None
            for key, variable in self.vars.items():
                if key != "animate_preview":
                    variable.set(getattr(self.p, key))
            for key in ("road_color", "stripe_color"):
                self._fix_btn_fg(getattr(self, f"{key}_btn"), getattr(self.p, key))
            self._update_background_label()
            self._refresh_pedestrians()
        finally:
            self._loading = False
        self.schedule()

    def _toggle_anim_preview_stop(self):
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self._anim_t = 0

    def add_shot(self):
        if self._rendering:
            return
        try:
            p = self._current_params()
        except (OSError, ValueError, TypeError, tk.TclError) as e:
            messagebox.showerror("Could not add shot", str(e), parent=self.root)
            return
        shot = Shot(p)
        self.sequence.shots.append(shot)
        self._refresh_cards()
        self.sequence_canvas.update_idletasks()
        self.sequence_canvas.xview_moveto(1)
        self._editor_changed()

    def edit_shot(self, shot_id):
        if self._rendering or not self._allow_leave_shot():
            return
        shot = next(shot for shot in self.sequence.shots if shot.id == shot_id)
        self._edit_id = shot.id
        self._load_editor(shot.params)
        self._editor_changed()

    def update_shot(self):
        shot = self._editing_shot()
        if self._rendering or shot is None:
            return False
        try:
            p = self._current_params()
        except (OSError, ValueError, TypeError, tk.TclError) as e:
            messagebox.showerror("Could not update shot", str(e), parent=self.root)
            return False
        shot.params = p
        self._thumbnails.pop(shot.id, None)
        self._refresh_cards()
        self._editor_changed()
        return True

    def done_editing(self):
        if not self._rendering and self._allow_leave_shot():
            self._edit_id = None
            self._update_edit_label()

    def remove_shot(self, shot_id):
        if self._rendering or not messagebox.askyesno(
                "Remove shot", "Remove this shot and any pending edits to it?",
                parent=self.root):
            return
        self.sequence.shots[:] = [shot for shot in self.sequence.shots if shot.id != shot_id]
        self._thumbnails.pop(shot_id, None)
        if self._edit_id == shot_id:
            self._edit_id = None
        self._refresh_cards()
        self._editor_changed()

    def _select_sequence(self, event=None):
        selected = self.project.sequences[self.sequence_choice.current()]
        if self._rendering or not self._allow_leave_shot():
            self.sequence_choice.current(self.project.sequences.index(self.sequence))
            return
        self._edit_id = None
        self.project.active_id = selected.id
        self._refresh_cards()
        self.sequence_canvas.xview_moveto(0)
        self._editor_changed()

    def _sequence_name(self, initial, exclude_id=None):
        name = simpledialog.askstring("Sequence name", "Sequence name (also the PNG prefix):",
                                      initialvalue=initial, parent=self.root)
        if name is None:
            return None
        try:
            validate_sequence_name(name)
            if any(s.name.casefold() == name.casefold() and s.id != exclude_id
                   for s in self.project.sequences):
                raise ValueError("A sequence with this name already exists.")
        except ValueError as e:
            messagebox.showerror("Invalid sequence name", str(e), parent=self.root)
            return None
        return name

    def add_sequence(self):
        if self._rendering or not self._allow_leave_shot():
            return
        index = 1
        while any(s.name.casefold() == f"sequence{index}" for s in self.project.sequences):
            index += 1
        name = self._sequence_name(f"Sequence{index}")
        if name is not None:
            sequence = Sequence(name)
            self.project.sequences.append(sequence)
            self.project.active_id = sequence.id
            self._edit_id = None
            self._refresh_sequences()
            self._editor_changed()

    def rename_sequence(self):
        if self._rendering:
            return
        name = self._sequence_name(self.sequence.name, self.sequence.id)
        if name is not None:
            self.sequence.name = name
            self._refresh_sequences()
            self._editor_changed()

    def delete_sequence(self):
        if self._rendering or not messagebox.askyesno(
                "Delete sequence", "Delete this sequence, all its shots, and pending shot edits?",
                parent=self.root):
            return
        for shot in self.sequence.shots:
            self._thumbnails.pop(shot.id, None)
        self.project.sequences.remove(self.sequence)
        if not self.project.sequences:
            self.project.sequences.append(Sequence("Sequence1"))
        self.project.active_id = self.project.sequences[0].id
        self._edit_id = None
        self._refresh_sequences()
        self._editor_changed()

    def save_project(self, save_as=False):
        if self._rendering:
            return False
        path = self.project_path
        if save_as or path is None:
            chosen = filedialog.asksaveasfilename(
                parent=self.root, defaultextension=".json",
                filetypes=[("CrossRoads project", "*.json")],
                initialfile=path.name if path else "CrossRoads.json")
            if not chosen:
                return False
            path = Path(chosen)
        try:
            state = self._project_state()
            save_project_file(self.project, path)
        except (OSError, ValueError, TypeError, tk.TclError) as e:
            messagebox.showerror("Could not save project", str(e), parent=self.root)
            return False
        self.project_path = Path(path)
        self._saved_state = state
        self._editor_changed()
        self.status.config(text=f"Project saved: {path}")
        return True

    def save_project_as(self):
        return self.save_project(save_as=True)

    def _allow_replace_project(self):
        try:
            dirty = self._project_state() != self._saved_state
        except (ValueError, TypeError, tk.TclError):
            dirty = True
        if not dirty:
            return True
        choice = messagebox.askyesnocancel(
            "Unsaved project", "Save project changes before continuing?\n"
            "The editor setup is saved separately; saved shots are not implicitly updated.",
            parent=self.root)
        return choice is False or (choice is True and self.save_project())

    def _install_project(self, project, path=None):
        self._project_revision += 1
        self._preview_revision += 1
        self._pending_preview = None
        self._pending_cache_clear = True
        if self._worker_cancel is not None:
            self._worker_cancel.set()
        self.project = project
        self.project_path = path
        self._edit_id = None
        self._thumbnails.clear()
        _road_texture.cache_clear()
        _tiled_road_texture.cache_clear()
        self._load_editor(project.editor)
        self._refresh_sequences()
        self.sequence_canvas.xview_moveto(0)
        self._saved_state = self._project_state()
        self._editor_changed()

    def new_project(self):
        if not self._rendering and self._allow_replace_project():
            self._install_project(new_project(Params()))

    def open_project(self):
        if self._rendering:
            return
        chosen = filedialog.askopenfilename(parent=self.root,
                                             filetypes=[("CrossRoads project", "*.json")])
        if not chosen:
            return
        path = Path(chosen)
        self._begin_operation(
            "load", partial(_background_call, load_project_file, (path,)), path)

    def close_project(self):
        if self._rendering:
            messagebox.showinfo("Task in progress", "Cancel the current task before closing.",
                                parent=self.root)
        elif self._allow_replace_project():
            for job in (self._job, self._anim_job):
                if job is not None:
                    self.root.after_cancel(job)
            self.root.destroy()

    def _set_render_controls(self, busy):
        if busy:
            def disable(widget):
                if widget == self.cancel_render_button:
                    return
                if "state" in widget.keys():
                    state = str(widget.cget("state"))
                    self._disabled_widgets.append((widget, state))
                    widget.configure(state="disabled")
                for child in widget.winfo_children():
                    disable(child)
            disable(self.root)
        else:
            for widget, state in self._disabled_widgets:
                if widget.winfo_exists():
                    widget.configure(state=state)
            self._disabled_widgets.clear()
        for index in range(4):
            self.project_menu.entryconfigure(index, state="disabled" if busy else "normal")
        self.cancel_render_button.config(state="normal" if busy else "disabled")
        self._update_edit_label()

    def render_sequence(self):
        if self._rendering:
            return
        if not self.sequence.shots:
            messagebox.showinfo("Empty sequence", "Add at least one shot first.", parent=self.root)
            return
        if not self._allow_leave_shot():
            return
        directory = filedialog.askdirectory(parent=self.root, title="Choose sequence output folder")
        if not directory:
            return
        snapshot = Sequence(self.sequence.name, [Shot(replace(s.params)) for s in self.sequence.shots])
        self._render_directory = Path(directory)
        self._render_written = 0
        self._render_total = sum(s.params.anim_cycles * crowd.FRAME_COUNT for s in snapshot.shots)
        self._begin_operation(
            "sequence", partial(_export_sequence, snapshot, Path(directory)))

    def cancel_sequence_render(self):
        if self._rendering:
            self._operation_cancel.set()
            if self._future is not None and self._work_kind in ("load", "sequence", "animation"):
                self._worker_cancel.set()
            self.cancel_render_button.config(state="disabled")
            self.status.config(text="Cancelling... waiting for the current operation to finish.")

    def _update_background_label(self):
        paths = self.p.background_images
        if not paths:
            text = "Default: BG.png"
        elif len(paths) == 1:
            text = Path(paths[0]).name
        else:
            text = f"{len(paths)} images selected (first: {Path(paths[0]).name})"
        self.background_label.config(text=text)

    def choose_backgrounds(self):
        paths = filedialog.askopenfilenames(
            parent=self.root, title="Choose background images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")])
        if not paths:
            return
        try:
            for path in paths:
                with Image.open(path) as image:
                    image.load()
        except (OSError, ValueError) as e:
            messagebox.showerror("Could not load background", str(e), parent=self.root)
            return
        _road_texture.cache_clear()
        _tiled_road_texture.cache_clear()
        self.p.background_images = tuple(paths)
        self._update_background_label()
        self.schedule()

    def reset_background(self):
        self.p.background_images = ()
        self._update_background_label()
        self.schedule()

    def _build_pedestrian_tab(self, tab):
        tab.columnconfigure(0, weight=1)
        self.pedestrian_list = tk.Listbox(tab, height=3, exportselection=False)
        self.pedestrian_list.grid(row=0, column=0, sticky="ew")
        self.pedestrian_list.bind("<<ListboxSelect>>", self._select_pedestrian)
        self.pedestrian_hint = tk.StringVar()
        self.pedestrian_buttons = []

        def buttons(row, actions, *, selected=True):
            bar = ttk.Frame(tab)
            bar.grid(row=row, column=0, sticky="w")
            created = []
            for label, command in actions:
                button = ttk.Button(bar, text=label, command=command)
                button.pack(side="left", padx=(0, 3))
                created.append(button)
                if selected or label == "Remove":
                    self.pedestrian_buttons.append(button)
            return created

        buttons(1, (("Add pedestrian", self._add_pedestrian),
                    ("Add imported…", self._add_side_pedestrian), ("Remove", self._remove_pedestrian)),
                selected=False)
        look = ttk.Frame(tab)
        look.grid(row=2, column=0, sticky="w", pady=4)
        self.part_buttons = []
        for i, label in enumerate(("Legs", "Arms", "Head", "Hair")):
            button = tk.Button(look, width=5, text=label, command=lambda i=i: self._pick_part(i))
            button.pack(side="left", padx=(0, 3))
            self.part_buttons.append(button)
            self.pedestrian_buttons.append(button)
        self.hair_variant = tk.BooleanVar()
        hair = ttk.Checkbutton(look, text="Female hair", variable=self.hair_variant,
                               command=lambda: self._change_pedestrian(female=self.hair_variant.get()))
        hair.pack(side="left")
        self.pedestrian_buttons.append(hair)
        self.builtin_look_buttons = [*self.part_buttons, hair]
        self.side_buttons = buttons(3, (("Relink walk…", self._relink_side_walk),
                                        ("Stride…", self._set_side_stride)))
        self.speed_button, _, _ = buttons(
            4, (("Speed…", self._set_pedestrian_speed), ("Size…", self._set_pedestrian_size),
                ("Draw route / place", self._start_route)))
        self.route_controls = ttk.Frame(tab)
        self.route_controls.grid(row=5, column=0, sticky="w")
        for label, command in (("Undo point", self._undo_route_point), ("Finish route", self._finish_route),
                               ("Cancel", self._cancel_route)):
            ttk.Button(self.route_controls, text=label, command=command).pack(side="left", padx=(0, 3))
        ttk.Label(tab, textvariable=self.pedestrian_hint, wraplength=295).grid(row=6, column=0, sticky="w")

        heading = ttk.Frame(tab)
        heading.grid(row=7, column=0, sticky="ew", pady=(6, 0))
        self.timeline_label = ttk.Label(heading)
        self.timeline_label.pack(side="left")
        ttk.Button(heading, text="Go to…", width=7, command=self._go_to_frame).pack(side="right")
        self.timeline = tk.Canvas(tab, height=76, width=295, background="#262626", highlightthickness=0)
        self.timeline.grid(row=8, column=0, sticky="ew")
        self.timeline.bind("<Configure>", lambda _: self._draw_timeline())
        self.timeline.bind("<Button-1>", self._timeline_press)
        self.timeline.bind("<B1-Motion>", self._timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self._timeline_release)
        self.timeline.bind("<Motion>", self._timeline_hover)
        self.timeline.bind("<Leave>", lambda _: self.timeline.configure(cursor=""))
        self._animate_checkbox(tab, 9)
        self.clip_list = tk.Listbox(tab, height=3, exportselection=False)
        self.clip_list.grid(row=10, column=0, sticky="ew")
        self.clip_buttons = buttons(11, (("Add clip…", self._add_clip), ("Edit / relink…", self._edit_clip),
                                         ("Remove clip", self._remove_clip)))
        self.vars["anim_cycles"].trace_add("write", self._refresh_timeline_settings)
        self.vars["anim_fps"].trace_add("write", self._refresh_timeline_settings)
        self.timeline_help = ttk.Label(tab, wraplength=295)
        self.timeline_help.grid(row=12, column=0, sticky="w")
        pet = ttk.LabelFrame(tab, text="Dog interaction", padding=4)
        pet.grid(row=13, column=0, sticky="ew", pady=4)
        self.pet_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(pet, text="Pet dog, then leave together", variable=self.pet_enabled_var,
                        command=self._toggle_pet).grid(row=0, column=0, columnspan=4, sticky="w")
        self.pet_duration_var = tk.StringVar(value="2")
        ttk.Label(pet, text="Pet seconds:").grid(row=1, column=0)
        duration = ttk.Entry(pet, textvariable=self.pet_duration_var, width=6)
        duration.grid(row=1, column=1)
        duration.bind("<Return>", self._set_pet_duration)
        ttk.Button(pet, text="Apply", command=self._set_pet_duration).grid(row=1, column=2)
        self.pet_position_button = ttk.Button(pet, text="Position petting pose",
                                              command=self._position_pet)
        self.pet_position_button.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(pet, text="Fit sequence", command=self._fit_interaction).grid(row=2, column=2)
        self.pet_step_var = tk.StringVar(value="1")
        ttk.Label(pet, text="Nudge (preview px):").grid(row=3, column=0, columnspan=2)
        ttk.Entry(pet, textvariable=self.pet_step_var, width=6).grid(row=3, column=2)
        nudges = ttk.Frame(pet)
        nudges.grid(row=4, column=0, columnspan=4, sticky="w")
        for label, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            ttk.Button(nudges, text=label, width=5,
                       command=lambda dx=dx, dy=dy: self._nudge_pet(dx, dy)).pack(side="left")
        ttk.Button(nudges, text="Reset", width=6, command=self._reset_pet_position).pack(side="left")
        self.pet_hint = ttk.Label(pet, wraplength=295)
        self.pet_hint.grid(row=5, column=0, columnspan=4, sticky="w")
        self.pet_controls = [widget for parent in (pet, nudges) for widget in parent.winfo_children()
                             if isinstance(widget, (ttk.Button, ttk.Checkbutton, ttk.Entry))]
        self._refresh_pedestrians()

    def _toggle_pet(self):
        if self._pedestrian() is None:
            self.pet_enabled_var.set(False)
            messagebox.showinfo("Dog interaction", "Select a custom pedestrian first.", parent=self.root)
            return
        self._change_pedestrian(pet_dog=self.pet_enabled_var.get())

    def _set_pet_duration(self, _event=None):
        try:
            seconds = float(self.pet_duration_var.get())
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Enter a positive, finite petting duration.")
        except ValueError as error:
            messagebox.showerror("Invalid petting duration", str(error), parent=self.root)
            return
        self._change_pedestrian(pet_seconds=seconds)

    def _selected_interaction(self):
        spec = self._pedestrian()
        if spec is None or not spec.pet_dog:
            return None
        return interaction_for(self.read_params())

    def _fit_interaction(self):
        interaction = self._selected_interaction()
        if interaction is not None:
            self.vars["anim_cycles"].set(max(self.p.anim_cycles, math.ceil((interaction.end + 1) / 12)))
            self.schedule()

    def _position_pet(self):
        if self._positioning_pet:
            self._positioning_pet = False
            self._pet_drag = None
        else:
            interaction = self._selected_interaction()
            if interaction is None:
                messagebox.showinfo("Position petting pose", "Enable the dog interaction first.", parent=self.root)
                return
            if interaction.pet_start >= self._timeline_frames():
                if not messagebox.askyesno("Petting outside shot", "Fit the sequence to show the petting pose?",
                                          parent=self.root):
                    return
                self._fit_interaction()
            if self._pedestrian().walk_duration is None:
                self._change_pedestrian(walk_duration=interaction.duration)
            self._route_draft = None
            self._positioning_pet = True
            self._scrub_to(interaction.pet_start)
        self._refresh_pedestrians()
        self.schedule()

    def _move_pet_pixels(self, dx, dy):
        interaction = self._selected_interaction()
        cam = self._preview_cam
        if interaction is None or cam is None:
            messagebox.showerror("Position unavailable", "Wait for a dog-interaction preview.", parent=self.root)
            return
        sx, sy = cam.project(cam.to_cam(*interaction.stop))
        target = cam.ground_point(sx + dx / self._preview_scale, sy + dy / self._preview_scale)
        if target is None or not all(math.isfinite(v) for v in target):
            messagebox.showerror("Invalid position", "That position is outside the visible ground.", parent=self.root)
            return
        spec = self._pedestrian()
        offset = tuple(spec.pet_offset[i] + target[i] - interaction.stop[i] for i in (0, 1))
        self._change_pedestrian(pet_offset=offset)
        if self._positioning_pet:
            updated = self._selected_interaction()
            self._scrub_to(updated.pet_start)

    def _drag_pet_position(self, event):
        if self._pet_drag is not None and not self._rendering and self._positioning_pet:
            x, y = self._pet_drag
            self._move_pet_pixels(event.x - x, event.y - y)
            self._pet_drag = event.x, event.y

    def _nudge_pet(self, dx, dy):
        if not self._positioning_pet:
            self._position_pet()
        if not self._positioning_pet:
            return
        try:
            step = float(self.pet_step_var.get())
            if not math.isfinite(step) or step <= 0:
                raise ValueError("Nudge size must be a positive number.")
        except ValueError as error:
            messagebox.showerror("Invalid nudge size", str(error), parent=self.root)
            return
        self._move_pet_pixels(dx * step, dy * step)

    def _reset_pet_position(self):
        self._change_pedestrian(pet_offset=(0.0, 0.0))
        if self._positioning_pet:
            self._scrub_to(self._selected_interaction().pet_start)

    def _pedestrian(self):
        return next((p for p in self.p.custom_pedestrians if p.id == self._pedestrian_id), None)

    def _refresh_pedestrians(self, clip_selection=None):
        spec = self._pedestrian()
        self.pedestrian_list.delete(0, "end")
        for i, person in enumerate(self.p.custom_pedestrians):
            kind = "Imported" if person.side_walk_folder is not None else "Pedestrian"
            suffix = " (in place)" if len(person.route) == 1 else "" if person.route else " (not placed)"
            self.pedestrian_list.insert("end", f"{kind} {i + 1}{suffix}")
            if person is spec:
                self.pedestrian_list.selection_set(i)
                self.pedestrian_list.see(i)
        for button in self.pedestrian_buttons:
            button.config(state="normal" if spec else "disabled")
        imported = spec is not None and spec.side_walk_folder is not None
        for button in (*self.builtin_look_buttons, *self.clip_buttons, *self.pet_controls):
            button.configure(state="normal" if spec and not imported else "disabled")
        for button in self.side_buttons:
            button.configure(state="normal" if imported else "disabled")
        if imported and len(spec.route) == 1:
            self.speed_button.configure(state="disabled")
            self.side_buttons[1].configure(state="disabled")
        self.hair_variant.set(spec.female if spec else False)
        self.pet_enabled_var.set(spec.pet_dog if spec else False)
        self.pet_duration_var.set(f"{spec.pet_seconds:g}" if spec else "2")
        self.pet_position_button.configure(text="Done positioning" if self._positioning_pet
                                           else "Position petting pose")
        for i, button in enumerate(self.part_buttons):
            self._fix_btn_fg(button, spec.colors[i] if spec else "#dddddd")
        self._refresh_timeline_settings(clip_selection=clip_selection)
        self._draw_route_overlay()

    def _refresh_route_hint(self):
        spec = self._pedestrian()
        if self._route_draft is None:
            self.route_controls.grid_remove()
            speed = (interaction_for(self.p, validate=False).speed if spec and spec.pet_dog
                     else spec.walk_speed(self.p.anim_fps) if spec else 0)
            stationary = spec is not None and spec.side_walk_folder is not None and len(spec.route) == 1
            text = (("" if stationary else f"Speed: {speed:.3g} units/sec. ")
                    + f"Size: {spec.height or self.p.person_size:g} units. "
                    + (f"Animation: {Path(spec.side_walk_folder).name}. "
                       + ("" if stationary else f"Stride: {spec.side_walk_stride:g} heights. ")
                       + ("[missing: relink] " if not Path(spec.side_walk_folder).is_dir() else "")
                       if spec.side_walk_folder is not None else "")
                    + ("Animating in place." if stationary else
                       f"Route: {len(spec.route)} points." if spec.route else
                       "Draw a route or place this imported character." if spec.side_walk_folder is not None else
                       "Draw a route to place this pedestrian.")
                    if spec else "No custom pedestrian selected. Use Add pedestrian to start.")
        else:
            self.route_controls.grid()
            text = ("Click the start on the preview." if not self._route_draft else
                    "Click more points for a route, or Finish route to animate in place."
                    if spec and spec.side_walk_folder is not None and len(self._route_draft) == 1 else
                    "Click waypoints and the stop on the preview, then Finish route.")
        self.pedestrian_hint.set(text)

    def _select_pedestrian(self, _event):
        selected = self.pedestrian_list.curselection()
        if not selected or self._rendering:
            return
        pid = self.p.custom_pedestrians[selected[0]].id
        if pid != self._pedestrian_id:
            was_positioning = self._positioning_pet
            self._pedestrian_id = pid
            self._route_draft = None
            self._positioning_pet = False
            self._pet_drag = None
            self._timeline_notice = ""
            self._refresh_pedestrians()
            if was_positioning:
                self.schedule()

    def _add_pedestrian(self):
        self._positioning_pet = False
        self._pet_drag = None
        spec = crowd.CustomPedestrian(uuid.uuid4().hex)
        self._pedestrian_id = spec.id
        self.p.custom_pedestrians += (spec,)
        self._route_draft = None
        self._refresh_pedestrians()
        self.schedule()

    def _choose_side_walk(self):
        folder = filedialog.askdirectory(parent=self.root, title="Choose numbered PNG walk-cycle folder")
        if not folder:
            return None
        folder = str(Path(folder).resolve())
        try:
            crowd.load_side_walk.cache_clear()
            crowd.load_side_walk(folder)
        except (OSError, ValueError) as error:
            messagebox.showerror("Invalid walk cycle", str(error), parent=self.root)
            return None
        self._pending_cache_clear = True
        return folder

    def _add_side_pedestrian(self):
        folder = self._choose_side_walk()
        if folder is not None:
            self._add_pedestrian()
            self._change_pedestrian(side_walk_folder=folder)

    def _relink_side_walk(self):
        spec = self._pedestrian()
        if spec is not None and spec.side_walk_folder is not None:
            folder = self._choose_side_walk()
            if folder is not None:
                self._change_pedestrian(side_walk_folder=folder)

    def _set_side_stride(self):
        spec = self._pedestrian()
        if spec is not None and spec.side_walk_folder is not None:
            stride = simpledialog.askfloat(
                "Walk-cycle stride", "Distance per cycle, in pedestrian heights:",
                parent=self.root, initialvalue=spec.side_walk_stride, minvalue=0.01)
            if stride is not None:
                if not math.isfinite(stride) or stride <= 0:
                    messagebox.showerror("Invalid stride", "Enter a positive, finite stride.", parent=self.root)
                else:
                    self._change_pedestrian(side_walk_stride=stride)

    def _remove_pedestrian(self):
        self._positioning_pet = False
        self._pet_drag = None
        self.p.custom_pedestrians = tuple(p for p in self.p.custom_pedestrians if p.id != self._pedestrian_id)
        self._pedestrian_id = None
        self._route_draft = None
        self._refresh_pedestrians()
        self.schedule()

    def _change_pedestrian(self, *, clip_selection=None, **changes):
        spec = self._pedestrian()
        if spec is not None:
            pedestrians = tuple(replace(p, **changes) if p is spec else p
                                for p in self.p.custom_pedestrians)
            try:
                params = replace(self.read_params(), custom_pedestrians=pedestrians)
                validate_project_params(params)
            except ValueError as error:
                messagebox.showerror("Invalid pedestrian settings", str(error), parent=self.root)
                self._refresh_pedestrians()
                return
            self.p.custom_pedestrians = pedestrians
            if not self._pedestrian().pet_dog:
                self._positioning_pet = False
                self._pet_drag = None
            self._refresh_pedestrians(clip_selection)
            self.schedule()

    def _pick_part(self, index):
        spec = self._pedestrian()
        if spec is not None:
            _, color = colorchooser.askcolor(spec.colors[index], parent=self.root)
            if color:
                self._change_pedestrian(colors=spec.colors[:index] + (color,) + spec.colors[index + 1:])

    def _set_pedestrian_speed(self):
        spec = self._pedestrian()
        if spec is not None:
            interaction = self._selected_interaction()
            speed = simpledialog.askfloat("Walking speed", "World units per second (snaps to whole frames):",
                                          parent=self.root,
                                          initialvalue=interaction.speed if interaction else spec.walk_speed(self.p.anim_fps),
                                          minvalue=0.01, maxvalue=100)
            if speed is not None and math.isfinite(speed):
                length = interaction.distances[-1] if interaction else spec.route_length()
                duration = max(2, math.ceil(length * self.p.anim_fps / speed) + 1)
                self._change_pedestrian(speed=speed, walk_duration=duration if spec.route else None)

    def _set_pedestrian_size(self):
        spec = self._pedestrian()
        if spec is not None:
            current_height = spec.height or self.read_params().person_size
            height = simpledialog.askfloat("Pedestrian size", "Height in world units:",
                                           parent=self.root, initialvalue=current_height,
                                           minvalue=MIN_PERSON_SIZE, maxvalue=MAX_PERSON_SIZE)
            if height is not None:
                if not math.isfinite(height):
                    messagebox.showerror("Invalid pedestrian size", "Enter a finite height.", parent=self.root)
                    return
                self._change_pedestrian(height=height)

    def _pedestrian_tab_active(self):
        return self.notebook.select() == str(self.pedestrian_tab)

    def _pedestrian_tab_changed(self, _event):
        if self._pedestrian_tab_active():
            self.read_params()
            self._refresh_route_hint()
        self._draw_route_overlay()
        if not self.vars["animate_preview"].get():
            self.schedule()

    def _start_route(self):
        if self._pedestrian() is not None:
            was_positioning = self._positioning_pet
            self._positioning_pet = False
            self._route_draft = []
            self._refresh_route_hint()
            self._draw_route_overlay()
            if was_positioning:
                self.schedule()

    def _place_route_point(self, event):
        if self._positioning_pet and not self._rendering and self._pedestrian_tab_active():
            self._pet_drag = event.x, event.y
            return
        if self._rendering or self._route_draft is None or not self._pedestrian_tab_active():
            return
        if self._preview_scale != 1.0:
            self.pedestrian_hint.set("Wait for the full shot preview before drawing a route.")
            return
        cam, bounds = self._preview_cam, self._preview_bounds
        if cam is None or bounds is None:
            self.pedestrian_hint.set("Wait for the shot preview before clicking.")
            return
        x0, y0, x1, y1 = bounds
        point = (cam.ground_point(event.x - x0, event.y - y0)
                 if x0 <= event.x < x1 and y0 <= event.y < y1 else None)
        if point is None or (self._route_draft and math.dist(self._route_draft[-1], point) < 1e-8):
            self.pedestrian_hint.set("Click a different point on the visible ground inside the shot.")
            return
        self._route_draft.append(point)
        self._refresh_route_hint()
        self._draw_route_overlay()

    def _undo_route_point(self):
        if self._route_draft:
            self._route_draft.pop()
            self._refresh_route_hint()
            self._draw_route_overlay()

    def _cancel_route(self):
        self._route_draft = None
        self._refresh_route_hint()
        self._draw_route_overlay()

    def _finish_route(self):
        spec = self._pedestrian()
        minimum = 1 if spec is not None and spec.side_walk_folder is not None else 2
        if self._route_draft is None or len(self._route_draft) < minimum:
            self.pedestrian_hint.set("Click a position on the preview." if minimum == 1 else
                                     "Place at least a start and a stop.")
            return
        route = tuple(self._route_draft)
        self._route_draft = None
        if spec is not None:
            duration = spec.walk_duration
            if len(route) == 1:
                duration = None
            elif len(spec.route) < 2:
                duration = replace(spec, route=route).walk_end(self.p.anim_fps) - spec.departure + 1
            self._change_pedestrian(route=route, walk_duration=duration)

    def _draw_route_overlay(self):
        self.canvas.delete("route")
        if self._positioning_pet or self._preview_scale != 1.0:
            return
        cam, bounds = self._preview_cam, self._preview_bounds
        spec = self._pedestrian()
        if cam is None or bounds is None or not self._pedestrian_tab_active() or spec is None:
            return
        editing = self._route_draft is not None
        route = self._route_draft if editing else spec.route
        if not editing and spec.pet_dog:
            route = interaction_for(self.p, validate=False).route
        previous = None
        for i, (x, y) in enumerate(route):
            point = cam.to_cam(x, y)
            if point[2] <= cam.near:
                previous = None
                continue
            sx, sy = cam.project(point)
            position = (sx + bounds[0], sy + bounds[1])
            if previous is not None:
                self.canvas.create_line(*previous, *position, fill="#ffdb6c", width=2, tags="route")
            previous = position
            if 0 <= sx < cam.w and 0 <= sy < cam.h:
                px, py = position
                self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4,
                                        fill="#ffdb6c", outline="#222", tags="route")
                label = ("Position" if not editing and len(route) == 1 else
                         "Start" if i == 0 else "Stop" if not editing and i == len(route) - 1 else str(i))
                self.canvas.create_text(px, py - 7, text=label, anchor="s", fill="#ffdb6c", tags="route")

    def _timeline_frames(self):
        return self.p.anim_cycles * crowd.FRAME_COUNT

    def _refresh_timeline_settings(self, *_args, clip_selection=None):
        if not self._loading:
            self._read_animation_settings(self.p)
        self._scrub_frame = min(self._scrub_frame, self._timeline_frames() - 1)
        if clip_selection is None:
            clip_selection = next(iter(self.clip_list.curselection()), None)
        self.clip_list.delete(0, "end")
        spec = self._pedestrian()
        if spec:
            for block in spec.clips:
                warning = (" [outside shot]" if block.end >= self._timeline_frames() else "")
                if not Path(block.folder).is_dir():
                    warning += " [missing: relink]"
                self.clip_list.insert("end", f"{block.start + 1}–{block.end + 1}: {Path(block.folder).name}{warning}")
            if clip_selection is not None and clip_selection < len(spec.clips):
                self.clip_list.selection_set(clip_selection)
        self._refresh_route_hint()
        if spec and spec.pet_dog:
            interaction = interaction_for(self.p, validate=False)
            warning = " [cut off: use Fit sequence]" if interaction.end >= self._timeline_frames() else ""
            self.pet_hint.configure(text=(
                f"Pet {interaction.pet_start + 1}-{interaction.return_start}. "
                f"Dog exits by frame {interaction.end + 1}.{warning}\n"
                + ("Drag preview to position; Done to finish. " if self._positioning_pet else "")
                + f"Offset: {spec.pet_offset[0]:.3f}, {spec.pet_offset[1]:.3f} world units."))
        else:
            self.pet_hint.configure(text="One pedestrian per shot. Draw a route and enable Show dog.")
        self._draw_timeline()

    def _timeline_span(self):
        return max(1, self.timeline.winfo_width() - self.TIMELINE_LEFT - 8)

    def _timeline_x(self, boundary):
        return self.TIMELINE_LEFT + self._timeline_span() * boundary / self._timeline_frames()

    def _timeline_blocks(self, spec):
        if spec is not None:
            if len(spec.route) >= 2:
                end = (interaction_for(self.p, validate=False).pet_start if spec.pet_dog
                       else spec.walk_end(self.p.anim_fps))
                yield "walk", None, spec.departure, end, "Approach" if spec.pet_dog else "Walk"
            for index, block in enumerate(spec.clips):
                yield "clip", index, block.start, block.end, Path(block.folder).name

    def _draw_timeline(self, frame=None):
        frame = self._scrub_frame if frame is None else frame
        size = self._timeline_frames()
        frame = min(frame, size - 1)
        canvas = self.timeline
        canvas.delete("all")
        for title, y in (("Scrub", 12), ("Walk", 34), ("Animation", 60)):
            canvas.create_text(4, y, text=title, fill="#ddd", anchor="w")
        left, right = self._timeline_x(0), self._timeline_x(size)
        canvas.create_line(left, 12, right, 12, fill="#999")
        for y0, y1 in self.TIMELINE_LANES.values():
            canvas.create_rectangle(left, y0, right, y1, fill="#383838", outline="")
        spec = self._pedestrian()
        label = f"Frame {frame + 1}/{size}"
        drag = self._block_drag
        for lane, index, start, end, name in self._timeline_blocks(spec):
            dragged = drag is not None and (lane, index) == (drag["lane"], drag["index"])
            if dragged:
                start, end = drag["range"]
            if lane == "walk":
                label += f"  |  Walk {start + 1}–{end + 1}"
                if end >= size:
                    label += " (outside)"
            x0, x1 = max(left, self._timeline_x(start)), min(right, self._timeline_x(end + 1))
            if x1 <= x0:
                continue
            y0, y1 = self.TIMELINE_LANES[lane]
            color = ("#d98278" if dragged and drag["error"] else
                     "#80c795" if lane == "walk" else "#8ab4e9")
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#1c1c1c")
            if x1 - x0 >= 10:
                for x in (x0 + 2, x1 - 2):
                    canvas.create_line(x, y0 + 5, x, y1 - 5, fill="#344454")
            if x1 - x0 >= 28:
                chars = max(1, int((x1 - x0 - 12) / 8))
                canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, fill="#151515", text=name[:chars])
        if spec is not None and spec.pet_dog:
            interaction = interaction_for(self.p, validate=False)
            for start, end, title, lane in (
                    (interaction.pet_start, interaction.return_start, "Pet", "clip"),
                    (interaction.return_start, interaction.end + 1, "Return + dog", "walk")):
                x0, x1 = max(left, self._timeline_x(start)), min(right, self._timeline_x(end))
                if x1 > x0:
                    y0, y1 = self.TIMELINE_LANES[lane]
                    canvas.create_rectangle(x0, y0, x1, y1, fill="#d4b37c", outline="")
                    canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=title, fill="#151515")
        playhead = self._timeline_x(frame + .5)
        canvas.create_line(playhead, 3, playhead, 72, fill="white")
        self.timeline_label.config(text=label)
        self.timeline_help.config(text=(drag["error"] if drag else "") or self._timeline_notice
                                  or ("The imported animation repeats throughout the shot."
                                      if spec and spec.side_walk_folder is not None and len(spec.route) == 1 else "")
                                  or "Drag a block to move it. Drag either edge to resize.")

    def _scrub_timeline(self, event):
        if self._rendering:
            return
        size = self._timeline_frames()
        self._scrub_to(max(0, min(size - 1, int((event.x - self.TIMELINE_LEFT)
                                               * size / self._timeline_span()))))

    def _timeline_hit(self, event):
        size = self._timeline_frames()
        for lane, index, start, end, _ in self._timeline_blocks(self._pedestrian()):
            y0, y1 = self.TIMELINE_LANES[lane]
            x0, x1 = self._timeline_x(start), self._timeline_x(min(end + 1, size))
            if y0 <= event.y <= y1 and x0 <= event.x < x1:
                edge = min(7, (x1 - x0) / 3)
                mode = "start" if event.x - x0 < edge else "end" if x1 - event.x <= edge else "move"
                return lane, index, start, end, mode
        return None

    def _timeline_hover(self, event):
        hit = self._timeline_hit(event)
        cursor = ("fleur" if hit[-1] == "move" else "sb_h_double_arrow") if hit else ""
        self.timeline.configure(cursor=cursor)

    def _timeline_press(self, event):
        if self._rendering:
            return
        self._timeline_notice = ""
        hit = self._timeline_hit(event)
        self._timeline_scrubbing = hit is None and event.y < self.TIMELINE_LANES["walk"][0]
        if hit:
            lane, index, start, end, mode = hit
            if self.vars["animate_preview"].get():
                self._scrub_to(self._preview_frame)
            if index is not None:
                self.clip_list.selection_clear(0, "end")
                self.clip_list.selection_set(index)
                self.clip_list.see(index)
            self._block_drag = dict(lane=lane, index=index, start=start, end=end, mode=mode,
                                    x=event.x, range=(start, end), error="")
            self._draw_timeline()
        elif self._timeline_scrubbing:
            self._scrub_timeline(event)

    def _timeline_drag(self, event):
        if self._rendering:
            return
        if self._timeline_scrubbing:
            self._scrub_timeline(event)
        drag = self._block_drag
        if drag is None:
            return
        size = self._timeline_frames()
        start, end = drag["start"], drag["end"]
        delta = round((event.x - drag["x"]) * size / self._timeline_span())
        mode = drag["mode"]
        if mode in ("start", "move"):
            start += delta
        if mode == "move":
            end += delta
        elif mode == "end" and delta:
            end = min(end, size - 1) + delta
        error = ""
        spec = self._pedestrian()
        if (start, end) != (drag["start"], drag["end"]) and spec is not None:
            minimum = 2 if drag["lane"] == "walk" else 1
            if start < 0 or end >= size:
                error = "Keep the block inside the shot."
            elif end - start + 1 < minimum:
                error = f"This block needs at least {minimum} frame(s)."
            elif drag["lane"] == "clip" and any(
                    i != drag["index"] and start <= block.end and block.start <= end
                    for i, block in enumerate(spec.clips)):
                error = "Animation clips on the same lane cannot overlap."
        drag.update(range=(start, end), error=error)
        self._draw_timeline()

    def _timeline_release(self, event):
        self._timeline_scrubbing = False
        if self._block_drag is None:
            return
        self._timeline_drag(event)
        drag, self._block_drag = self._block_drag, None
        self._timeline_notice = drag["error"]
        start, end = drag["range"]
        spec = self._pedestrian()
        if (self._rendering or spec is None or drag["error"]
                or (start, end) == (drag["start"], drag["end"])):
            self._draw_timeline()
            return
        if drag["lane"] == "walk":
            duration = spec.walk_duration if drag["mode"] == "move" else end - start + 1
            self._change_pedestrian(departure=start, walk_duration=duration)
        else:
            blocks = list(spec.clips)
            index = drag["index"]
            block = blocks[index] = replace(blocks[index], start=start, end=end)
            blocks.sort(key=lambda clip: clip.start)
            self._change_pedestrian(clips=tuple(blocks), clip_selection=blocks.index(block))

    def _go_to_frame(self):
        size = self._timeline_frames()
        frame = simpledialog.askinteger("Go to frame", f"Frame (1–{size}):", parent=self.root,
                                        initialvalue=self._scrub_frame + 1, minvalue=1, maxvalue=size)
        if frame is not None:
            self._scrub_to(frame - 1)

    def _scrub_to(self, frame):
        self.vars["animate_preview"].set(False)
        self._toggle_anim_preview_stop()
        self._scrub_frame = frame
        self._draw_timeline()
        self.schedule()

    def _change_clip(self, index):
        spec = self._pedestrian()
        if spec is None:
            return
        blocks = list(spec.clips)
        block = (blocks.pop(index) if index is not None else
                 crowd.ClipBlock("", self._scrub_frame, self._scrub_frame))
        folders = tuple(animation_folders(self.p))
        result = ClipDialog(self.root, folders, block, self._timeline_frames(), blocks).result
        if result is not None:
            blocks.append(result)
            blocks.sort(key=lambda clip: clip.start)
            self._change_pedestrian(clips=tuple(blocks), clip_selection=blocks.index(result))

    def _add_clip(self):
        self._change_clip(None)

    def _edit_clip(self):
        selected = self.clip_list.curselection()
        if selected:
            self._change_clip(selected[0])

    def _remove_clip(self):
        spec, selected = self._pedestrian(), self.clip_list.curselection()
        if spec is not None and selected:
            i = selected[0]
            self._change_pedestrian(clips=spec.clips[:i] + spec.clips[i + 1:])

    def _build_crowd_tab(self, tab):
        r = 0
        ev = tk.BooleanVar(value=self.p.crowd_enabled)
        self.vars["crowd_enabled"] = ev
        ttk.Checkbutton(tab, text="Show crowd", variable=ev, command=self.schedule).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r += 1
        shadow = tk.BooleanVar(value=self.p.shadows_enabled)
        self.vars["shadows_enabled"] = shadow
        ttk.Checkbutton(tab, text="Show character shadows", variable=shadow,
                        command=self.schedule).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        style = tk.StringVar(value=self.p.shadow_style)
        self.vars["shadow_style"] = style
        styles = ttk.Frame(tab)
        styles.grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r += 1
        for value, label in (("contact", "Contact"), ("cast", "Cast")):
            ttk.Radiobutton(styles, text=label, value=value, variable=style,
                            command=self.schedule).pack(side="left", padx=(0, 8))

        sliders = [
            ("crowd_count", "People", 0, 100, int),
            ("person_size", "Person size (world units)", MIN_PERSON_SIZE, MAX_PERSON_SIZE, float),
            ("size_var", "Size variation (±)", 0, 0.4, float),
            ("crowd_speed", "Walk speed", 0.25, 3, float),
        ]
        for key, label, lo, hi, typ in sliders:
            r = self._slider(tab, r, key, label, lo, hi, typ)

        looping = tk.BooleanVar(value=self.p.loop_pedestrians)
        self.vars["loop_pedestrians"] = looping
        ttk.Checkbutton(
            tab, text="Loop pedestrians",
            variable=looping, command=self.schedule,
        ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))
        r += 1

        seed_f = ttk.Frame(tab)
        seed_f.grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 0)); r += 1
        ttk.Label(seed_f, text="Seed").pack(side="left")
        sv = tk.IntVar(value=self.p.crowd_seed)
        self.vars["crowd_seed"] = sv
        ttk.Spinbox(seed_f, from_=0, to=999999, width=8, textvariable=sv, command=self.schedule).pack(side="left", padx=(4, 8))
        sv.trace_add("write", lambda *a: self.schedule())
        ttk.Button(seed_f, text="Reroll", width=7, command=lambda: sv.set(random.randrange(1000000))).pack(side="left")

        anim_f = ttk.Frame(tab)
        anim_f.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 0)); r += 1
        ttk.Label(anim_f, text="FPS").pack(side="left")
        fv = tk.IntVar(value=self.p.anim_fps)
        self.vars["anim_fps"] = fv
        ttk.Spinbox(anim_f, from_=1, to=60, width=4, textvariable=fv).pack(side="left", padx=(4, 10))
        ttk.Label(anim_f, text="Cycles").pack(side="left")
        cv = tk.IntVar(value=self.p.anim_cycles)
        self.vars["anim_cycles"] = cv
        ttk.Spinbox(anim_f, from_=1, to=500, width=4, textvariable=cv).pack(side="left", padx=4)
        ttk.Label(tab, text="(12 frames per walk cycle)", foreground="#666").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._animate_checkbox(tab, r)

    def _animate_checkbox(self, tab, row):
        ttk.Checkbutton(tab, text="▶ Animate preview", variable=self.vars["animate_preview"],
                        command=self._toggle_anim_preview).grid(
                            row=row, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def _build_dog_tab(self, tab):
        r = 0
        ev = tk.BooleanVar(value=self.p.dog_enabled)
        self.vars["dog_enabled"] = ev
        ttk.Checkbutton(tab, text="Show dog (image center)", variable=ev, command=self.schedule).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r += 1
        blink = tk.BooleanVar(value=self.p.dog_blink)
        self.vars["dog_blink"] = blink
        ttk.Checkbutton(tab, text="Dog blinks occasionally", variable=blink,
                        command=self.schedule).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r += 1
        sliders = [
            ("dog_size", "Dog size (world units)", 0.2, 3, float),
            ("dog_clearance", "People keep away (radius)", 0, 3, float),
            ("dog_tilt", "Head-tracking lean (°)", 0, 30, float),
            ("dog_hold", "Frames per dog / road drawing", 1, 6, int),
        ]
        for key, label, lo, hi, typ in sliders:
            r = self._slider(tab, r, key, label, lo, hi, typ)

    def _slider(self, parent, r, key, label, lo, hi, typ):
        ttk.Label(parent, text=label).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        v = tk.DoubleVar(value=getattr(self.p, key))
        self.vars[key] = v
        val_lbl = ttk.Label(parent, width=6, anchor="e")

        def on_change(*_):
            x = v.get()
            val_lbl.config(text=str(int(round(x))) if typ is int else f"{x:.2f}")
            self.schedule()

        ttk.Scale(parent, from_=lo, to=hi, variable=v, length=220).grid(row=r, column=0, columnspan=2, sticky="ew")
        val_lbl.grid(row=r, column=2, sticky="e")
        v.trace_add("write", on_change)
        on_change()
        return r + 1

    @staticmethod
    def _fix_btn_fg(btn, color):
        rgb = btn.winfo_rgb(color)
        lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 65535
        btn.config(bg=color, fg="black" if lum > 0.5 else "white", activebackground=color)

    def pick_color(self, key):
        c = colorchooser.askcolor(getattr(self.p, key), title=f"Choose {key.replace('_', ' ')}")[1]
        if c:
            setattr(self.p, key, c)
            self._fix_btn_fg(getattr(self, f"{key}_btn"), c)
            self.schedule()

    def read_params(self):
        p = self.p
        try:
            p.width = max(16, int(self.vars["width"].get()))
            p.height = max(16, int(self.vars["height"].get()))
        except (tk.TclError, ValueError):
            pass
        p.mode = self.vars["mode"].get()
        for key in ("angle", "zoom", "tilt", "camera_shift", "stripe_width", "stripe_gap", "depth",
                    "stripe_wobble", "background_zoom"):
            setattr(p, key, float(self.vars[key].get()))
        p.background_animated = bool(self.vars["background_animated"].get())
        p.stripes_animated = bool(self.vars["stripes_animated"].get())
        p.background_perspective = bool(self.vars["background_perspective"].get())
        p.stripe_count = int(round(self.vars["stripe_count"].get()))
        p.crowd_enabled = bool(self.vars["crowd_enabled"].get())
        p.loop_pedestrians = bool(self.vars["loop_pedestrians"].get())
        p.shadows_enabled = bool(self.vars["shadows_enabled"].get())
        p.shadow_style = self.vars["shadow_style"].get()
        p.crowd_count = int(round(self.vars["crowd_count"].get()))
        for key in ("person_size", "size_var", "crowd_speed"):
            setattr(p, key, float(self.vars[key].get()))
        p.dog_enabled = bool(self.vars["dog_enabled"].get())
        p.dog_blink = bool(self.vars["dog_blink"].get())
        for key in ("dog_size", "dog_clearance", "dog_tilt"):
            setattr(p, key, float(self.vars[key].get()))
        p.dog_hold = max(1, int(round(self.vars["dog_hold"].get())))
        self._read_animation_settings(p)
        return p

    def _read_animation_settings(self, p):
        for key in ("crowd_seed", "anim_fps", "anim_cycles"):
            try:
                setattr(p, key, max(0, int(self.vars[key].get())))
            except (tk.TclError, ValueError):
                pass
        p.anim_fps = max(1, p.anim_fps)
        p.anim_cycles = max(1, p.anim_cycles)

    def schedule(self):
        if self._rendering:
            return
        self._preview_revision += 1
        self._pending_preview = None
        if self._job:
            self.root.after_cancel(self._job)
        self._job = self.root.after(60, self.update_preview)
        if hasattr(self, "edit_label"):
            self._editor_changed()

    def update_preview(self):
        self._job = None
        if self._rendering:
            return
        p = replace(self.read_params())
        frame = (self._anim_t if self.vars["animate_preview"].get()
                 else self._scrub_frame if self._pedestrian_tab_active() else 0)
        if p.custom_pedestrians or self._pedestrian_tab_active():
            frame = min(frame, self._timeline_frames() - 1)
        cw = max(50, self.canvas.winfo_width())
        ch = max(50, self.canvas.winfo_height())
        s = min(cw / p.width, ch / p.height, 1.0)
        pw, ph = max(1, int(p.width * s)), max(1, int(p.height * s))
        self._preview_revision += 1
        preview = (_pet_preview_images if self._positioning_pet and self._pedestrian_tab_active()
                   else _preview_images)
        self._pending_preview = (
            "preview", partial(_background_call, preview, (p, pw, ph, frame)),
            (self._preview_revision, p, pw, ph, frame, cw, ch))
        self._ensure_worker_poll()

    def _stop_preview_on_error(self, error):
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self.vars["animate_preview"].set(False)
        self._preview_bg = None
        self.status.config(text="Preview failed.")
        messagebox.showerror("Preview failed", str(error), parent=self.root)

    @staticmethod
    def _crowd_on(p):
        return p.crowd_enabled and p.crowd_count > 0

    def _compose_crowd(self, bg: Image.Image, p: Params, cam: Camera, frame: int) -> Image.Image:
        return ShotRenderer(p, cam.w, cam.h).compose(bg, frame)

    def _toggle_anim_preview(self):
        if self.vars["animate_preview"].get():
            if self._anim_job is None:
                if self._pedestrian_tab_active():
                    self._anim_t = self._scrub_frame
                    self.update_preview()
                self._anim_tick()
        else:
            if self._anim_job is not None:
                self.root.after_cancel(self._anim_job)
                self._anim_job = None
            if self._pedestrian_tab_active():
                self._scrub_frame = min(self._preview_frame, self._timeline_frames() - 1)
                self._draw_timeline()
            self._anim_t = 0
            if self._job is not None:
                self.root.after_cancel(self._job)
            self.update_preview()

    def _anim_tick(self):
        self._anim_job = None
        if not self.vars["animate_preview"].get():
            return
        p = self.read_params()
        delay = 15
        if (not self._rendering and self._job is None and self._future is None
                and self._pending_preview is None):
            self._anim_t += 1
            if p.custom_pedestrians or self._pedestrian_tab_active():
                self._anim_t %= self._timeline_frames()
                self._scrub_frame = self._anim_t
                self._draw_timeline()
            self.update_preview()
            delay = max(1, round(1000 / p.anim_fps))
        self._anim_job = self.root.after(delay, self._anim_tick)

    def save(self):
        p = replace(self.read_params())
        frame = self._preview_frame
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")],
                                            initialfile=f"crosswalk_{p.mode}_{int(p.angle)}deg.png")
        if not path:
            return
        self.root.config(cursor="watch"); self.root.update()
        try:
            validate_animation_folders(p)
            bg = render(p, frame=frame)
            self._compose_crowd(bg, p, Camera(p, p.width, p.height), frame).save(path)
            self.status.config(text=f"Saved {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Save failed", str(e))
        finally:
            self.root.config(cursor="")

    def save_animation(self):
        if self._rendering:
            return
        p = replace(self.read_params())
        if not (self._crowd_on(p) or any(spec.route for spec in p.custom_pedestrians) or p.dog_enabled or
                (p.stripe_wobble > 0 and (p.background_animated or p.stripes_animated)) or
                (p.background_animated and len(p.background_images) > 1)):
            messagebox.showinfo("Nothing to animate", "Place a custom pedestrian, enable the crowd or dog, or animate the background or stripes.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".gif", filetypes=[("GIF preview + PNG frames", "*.gif")],
            initialfile=f"crosswalk_crowd_seed{p.crowd_seed}.gif")
        if not path:
            return
        base = path[:-4] if path.lower().endswith(".gif") else path
        self._render_directory = Path(base).parent
        self._animation_base = base
        self._render_written = 0
        self._render_total = p.anim_cycles * crowd.FRAME_COUNT
        self._begin_operation("animation", partial(_export_animation, p, base))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
