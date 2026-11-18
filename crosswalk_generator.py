"""3/4 overhead crosswalk image generator.

Run:  python crosswalk_generator.py

Stripe wobble controls both stripe jitter and the motion of the BG.png road
texture. Both use the Dog tab's frame hold, even when the dog is hidden.
The texture snaps to a new angle each drawing. Background zoom changes its
scale without affecting the camera. Below 1.0, blended crops use softer
contrast and fresh grain on each redraw; at 1.0 and above, the original is used.
Save PNG renders the displayed animation frame at the selected output size.
Pedestrians enter and exit outside the frame. GIFs with pedestrians play once
so restarting a short capture doesn't reset walkers in view.
"""

import math
import random
import time
import tkinter as tk
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageTk

import crowd
import dog as dog_mod

HFOV_DEG = 40.0  # horizontal field of view of the virtual camera
MAX_BARS_PER_BAND = 4000
ROAD_TEXTURE = Path(__file__).with_name("BG.png")
BACKGROUND_TILES = 5
MIN_BACKGROUND_ZOOM = 1.0 / BACKGROUND_TILES
MAX_BACKGROUND_ZOOM = 2.0
BACKGROUND_TURN_DEG = 110


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
    person_size: float = 1.75  # world-unit height (stripe width is ~0.5)
    size_var: float = 0.12  # +/- fraction of person_size
    crowd_speed: float = 1.0
    crowd_seed: int = 1
    anim_fps: int = 12
    anim_cycles: int = 4  # animation length in 12-frame walk cycles
    # dog (sits at the center of the image)
    dog_enabled: bool = True
    dog_size: float = 0.8  # world-unit height
    dog_clearance: float = 0.9  # radius people steer around
    dog_tilt: float = 12.0  # max lean in degrees when looking sideways
    dog_hold: int = 2  # output frames per dog drawing
    stripe_wobble: float = 0.35  # fraction of the allowed redraw displacement
    background_zoom: float = 1.0  # relative to the single-image cover scale


def make_dog(p: Params):
    if not p.dog_enabled:
        return None
    return dog_mod.Dog(height=p.dog_size, clearance=p.dog_clearance,
                       max_tilt=p.dog_tilt, hold=p.dog_hold)


# ---------------------------------------------------------------- camera


class Camera:
    def __init__(self, p: Params, width: int, height: int):
        self.w, self.h = width, height
        t = math.radians(p.tilt)
        half = math.tan(math.radians(HFOV_DEG) / 2)
        self.dist = (p.zoom / 2) / half
        self.focal = (width / 2) / half
        self.pos = (0.0, -self.dist * math.sin(t), self.dist * math.cos(t))
        self.fwd = (0.0, math.sin(t), -math.cos(t))
        self.up = (0.0, math.cos(t), math.sin(t))
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


@lru_cache(maxsize=1)
def _road_texture(road_color: str) -> Image.Image:
    with Image.open(ROAD_TEXTURE) as source:
        texture = source.convert("RGBA")
    # The flecks in BG.png are cutouts; a dark undercoat makes them visible.
    rgb = ImageColor.getcolor(road_color, "RGB")
    undercoat = (rgb[0] // 4, rgb[1] // 4, rgb[2] // 4)
    road = Image.new("RGB", texture.size, undercoat)
    road.paste(texture, mask=texture.getchannel("A"))
    return road


@lru_cache(maxsize=1)
def _tiled_road_texture(road_color: str) -> Image.Image:
    source = _road_texture(road_color).copy()
    source.thumbnail((1250, 1250), Image.Resampling.LANCZOS)
    average = source.resize((1, 1), Image.Resampling.BOX).resize(source.size)
    source = Image.blend(average, source, 0.18)
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
            patch = source.crop((x, y, x + size, y + size)).rotate(90 * rng.randrange(4))
            tiled.paste(patch, (col * pitch, row * pitch), masks[(col > 0) + 2 * (row > 0)])
    return tiled


def _draw_road(p: Params, width: int, height: int, drawing: int) -> Image.Image:
    zoom = min(MAX_BACKGROUND_ZOOM, max(MIN_BACKGROUND_ZOOM, p.background_zoom))
    if zoom < 1.0:
        texture = _tiled_road_texture(p.road_color)
        tiles_across = BACKGROUND_TILES
    else:
        texture = _road_texture(p.road_color)
        tiles_across = 1
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
    if zoom >= 1.0:
        return road

    # Grain covers the whole frame, so it doesn't follow the tile boundaries.
    grain_size = (max(1, width // 3), max(1, height // 3))
    grain = Image.frombytes("L", grain_size, rng.randbytes(math.prod(grain_size)))
    grain = grain.point([108 + round(i * 40 / 255) for i in range(256)])
    grain = grain.resize(road.size, Image.Resampling.BICUBIC)
    return ImageChops.add(road, grain.convert("RGB"), offset=-128)


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
    drawing = _stripe_drawing(p, frame)
    img = _draw_road(p, sw, shh, drawing)
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
        polygon, strokes = _sketch_stripe(p, along_x, start, end, unit, drawing)
        for points, light in strokes:
            _draw_stroke(draw, cam, points, along_x, pencil[light],
                         max(1, stroke_width // 2) if light else stroke_width)
        pts = _project_polygon(cam, polygon, along_x)
        if len(pts) >= 3:
            draw.polygon(pts, fill=p.stripe_color)
    if supersample > 1:
        img = img.resize((width, height), Image.LANCZOS)
    return img


# -------------------------------------------------------------------- UI


class App:
    PREVIEW_MAX = (900, 650)

    def __init__(self, root):
        self.root = root
        root.title("Crosswalk Generator")
        self.p = Params()
        self._job = None
        self._photo = None
        self._people = None
        self._people_key = None
        self._preview_bg = None
        self._preview_cam = None
        self._preview_drawing = None
        self._preview_frame = 0
        self._anim_t = 0
        self._anim_job = None

        ctrl = ttk.Frame(root, padding=10)
        ctrl.grid(row=0, column=0, sticky="ns")
        self.canvas = tk.Canvas(root, width=self.PREVIEW_MAX[0], height=self.PREVIEW_MAX[1], bg="#222", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda e: self.schedule())

        self.vars = {}

        nb = ttk.Notebook(ctrl)
        nb.grid(row=0, column=0, sticky="nsew")
        walk_tab = ttk.Frame(nb, padding=8)
        crowd_tab = ttk.Frame(nb, padding=8)
        dog_tab = ttk.Frame(nb, padding=8)
        nb.add(walk_tab, text="Crosswalk")
        nb.add(crowd_tab, text="Crowd")
        nb.add(dog_tab, text="Dog")
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
            ("zoom", "Zoom out (visible width)", 2, 100, float),
            ("tilt", "Camera tilt (°)", 0, 70, float),
            ("stripe_count", "Stripe count (0 = full)", 0, 40, int),
            ("stripe_width", "Stripe width", 0.05, 3, float),
            ("stripe_gap", "Stripe gap", 0.05, 3, float),
            ("depth", "Crosswalk depth", 0.5, 15, float),
            ("stripe_wobble", "Stripe / texture wobble (0 = still)", 0, 1, float),
            ("background_zoom", "Background zoom (tiles below 1.00)", MIN_BACKGROUND_ZOOM, MAX_BACKGROUND_ZOOM, float),
        ]
        for key, label, lo, hi, typ in sliders:
            r = self._slider(walk_tab, r, key, label, lo, hi, typ)

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

        ttk.Button(ctrl, text="Save PNG…", command=self.save).grid(row=1, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(ctrl, text="Save Animation…", command=self.save_animation).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.status = ttk.Label(ctrl, text="", foreground="#666", wraplength=260)
        self.status.grid(row=3, column=0, sticky="w", pady=(6, 0))

        self.schedule()

    def _build_crowd_tab(self, tab):
        r = 0
        ev = tk.BooleanVar(value=self.p.crowd_enabled)
        self.vars["crowd_enabled"] = ev
        ttk.Checkbutton(tab, text="Show crowd", variable=ev, command=self.schedule).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r += 1

        sliders = [
            ("crowd_count", "People", 0, 100, int),
            ("person_size", "Person size (world units)", 0.5, 4, float),
            ("size_var", "Size variation (±)", 0, 0.4, float),
            ("crowd_speed", "Walk speed", 0.25, 3, float),
        ]
        for key, label, lo, hi, typ in sliders:
            r = self._slider(tab, r, key, label, lo, hi, typ)

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
        ttk.Spinbox(anim_f, from_=1, to=40, width=4, textvariable=cv).pack(side="left", padx=4)
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

        ttk.Scale(parent, from_=lo, to=hi, variable=v, length=220, command=lambda _=None: on_change()).grid(row=r, column=0, columnspan=2, sticky="ew")
        val_lbl.grid(row=r, column=2, sticky="e")
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
        for key in ("angle", "zoom", "tilt", "stripe_width", "stripe_gap", "depth",
                    "stripe_wobble", "background_zoom"):
            setattr(p, key, float(self.vars[key].get()))
        p.stripe_count = int(round(self.vars["stripe_count"].get()))
        p.crowd_enabled = bool(self.vars["crowd_enabled"].get())
        p.crowd_count = int(round(self.vars["crowd_count"].get()))
        for key in ("person_size", "size_var", "crowd_speed"):
            setattr(p, key, float(self.vars[key].get()))
        p.dog_enabled = bool(self.vars["dog_enabled"].get())
        for key in ("dog_size", "dog_clearance", "dog_tilt"):
            setattr(p, key, float(self.vars[key].get()))
        p.dog_hold = max(1, int(round(self.vars["dog_hold"].get())))
        for key in ("crowd_seed", "anim_fps", "anim_cycles"):
            try:
                setattr(p, key, max(0, int(self.vars[key].get())))
            except (tk.TclError, ValueError):
                pass
        p.anim_fps = max(1, p.anim_fps)
        p.anim_cycles = max(1, p.anim_cycles)
        return p

    def _get_people(self, p, cam, dog):
        routes = walk_routes(p)
        obstacle = dog.obstacle() if dog else None
        key = (p.crowd_count, p.person_size, p.size_var, p.crowd_speed,
               p.crowd_seed, p.angle, p.tilt, p.zoom, cam.w, cam.h, obstacle, routes)
        if key != self._people_key:
            self._people = crowd.make_crowd(p, routes, cam, obstacle)
            self._people_key = key
        return self._people

    def schedule(self):
        if self._job:
            self.root.after_cancel(self._job)
        self._job = self.root.after(60, self.update_preview)

    def update_preview(self):
        self._job = None
        p = self.read_params()
        cw = max(50, self.canvas.winfo_width())
        ch = max(50, self.canvas.winfo_height())
        s = min(cw / p.width, ch / p.height, 1.0)
        pw, ph = max(1, int(p.width * s)), max(1, int(p.height * s))
        bg = render(p, pw, ph, supersample=2, frame=self._anim_t)
        self._preview_bg = bg
        self._preview_drawing = _stripe_drawing(p, self._anim_t)
        self._preview_cam = Camera(p, pw, ph)
        img = self._compose_crowd(bg, p, self._preview_cam, self._anim_t)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)
        self._preview_frame = self._anim_t
        self.status.config(text=f"Preview {pw}×{ph}  (output {p.width}×{p.height})")

    @staticmethod
    def _crowd_on(p):
        return p.crowd_enabled and p.crowd_count > 0

    def _compose_crowd(self, bg: Image.Image, p: Params, cam: Camera, frame: int) -> Image.Image:
        crowd_on = self._crowd_on(p)
        if not (crowd_on or p.dog_enabled):
            return bg
        img = bg.copy()
        dog = make_dog(p)
        people = self._get_people(p, cam, dog) if crowd_on else []
        crowd.draw_frame(img, cam, people, frame, dog)
        return img

    def _toggle_anim_preview(self):
        if self.vars["animate_preview"].get():
            if self._anim_job is None:
                self._anim_tick()
        elif self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None

    def _anim_tick(self):
        started = time.monotonic()
        self._anim_job = None
        if not self.vars["animate_preview"].get():
            return
        p = self.read_params()
        self._anim_t += 1
        cam = self._preview_cam
        if self._preview_bg is not None and cam is not None and self._job is None:
            drawing = _stripe_drawing(p, self._anim_t)
            if drawing != self._preview_drawing:
                self._preview_bg = render(p, *self._preview_bg.size, supersample=2,
                                          frame=self._anim_t)
                self._preview_drawing = drawing
            img = self._compose_crowd(self._preview_bg, p, cam, self._anim_t)
            self._photo = ImageTk.PhotoImage(img)
            cw = max(50, self.canvas.winfo_width())
            ch = max(50, self.canvas.winfo_height())
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo)
            self._preview_frame = self._anim_t
        elapsed_ms = (time.monotonic() - started) * 1000
        self._anim_job = self.root.after(max(1, round(1000 / p.anim_fps - elapsed_ms)),
                                         self._anim_tick)

    def save(self):
        p = replace(self.read_params())
        frame = self._preview_frame
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")],
                                            initialfile=f"crosswalk_{p.mode}_{int(p.angle)}deg.png")
        if not path:
            return
        self.root.config(cursor="watch"); self.root.update()
        try:
            bg = render(p, frame=frame)
            self._compose_crowd(bg, p, Camera(p, p.width, p.height), frame).save(path)
            self.status.config(text=f"Saved {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Save failed", str(e))
        finally:
            self.root.config(cursor="")

    def save_animation(self):
        p = replace(self.read_params())
        if not (self._crowd_on(p) or p.dog_enabled or p.stripe_wobble > 0):
            messagebox.showinfo("Nothing to animate", "Increase wobble or enable the crowd or dog.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".gif", filetypes=[("GIF preview + PNG frames", "*.gif")],
            initialfile=f"crosswalk_crowd_seed{p.crowd_seed}.gif")
        if not path:
            return
        base = path[:-4] if path.lower().endswith(".gif") else path
        total = p.anim_cycles * crowd.FRAME_COUNT
        self.root.config(cursor="watch")
        try:
            bg = render(p)
            drawing = _stripe_drawing(p, 0)
            cam = Camera(p, p.width, p.height)
            dog = make_dog(p)
            people = self._get_people(p, cam, dog) if self._crowd_on(p) else []
            gif_scale = min(1.0, 640 / p.width)
            gif_size = (max(1, int(p.width * gif_scale)), max(1, int(p.height * gif_scale)))
            gif_frames = []
            for t in range(total):
                next_drawing = _stripe_drawing(p, t)
                if next_drawing != drawing:
                    bg = render(p, frame=t)
                    drawing = next_drawing
                frame = bg.copy()
                crowd.draw_frame(frame, cam, people, t, dog)
                frame.save(f"{base}_{t + 1:04d}.png")
                small = frame.resize(gif_size, Image.LANCZOS) if gif_scale < 1.0 else frame
                gif_frames.append(small.convert("P", palette=Image.ADAPTIVE, colors=256))
                self.status.config(text=f"Rendering frame {t + 1}/{total}…")
                self.root.update()
            # Replaying a finite capture would jump pedestrians back to frame one.
            loop = {} if people else {"loop": 0}
            gif_frames[0].save(
                base + ".gif", save_all=True, append_images=gif_frames[1:],
                duration=int(round(1000 / p.anim_fps)), **loop)
            playback = " (plays once)" if people else ""
            self.status.config(text=f"Saved {total} PNG frames + {base}.gif{playback}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Save failed", str(e))
        finally:
            self.root.config(cursor="")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
