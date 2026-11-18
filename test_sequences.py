import json
import queue
import tempfile
import threading
import time
import tkinter as tk
import unittest
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageTk

import crosswalk_generator as app


class _ThreadWorker:
    """Keep delayed-job tests deterministic; integration tests use the real process."""

    def __init__(self):
        self.cancelled = threading.Event()
        self.progress = queue.Queue(maxsize=1)
        self.pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, function, token):
        return self.pool.submit(app._run_job, function, self.cancelled, self.progress, token)

    def shutdown(self, *, wait=False, cancel_futures=True):
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.image = self.directory / "road.png"
        Image.new("RGB", (20, 20), "red").save(self.image)
        self.params = app.Params(width=16, height=16, crowd_enabled=False,
                                 dog_enabled=False, stripe_wobble=0, anim_cycles=1,
                                 background_images=(str(self.image),))
        self.path = self.directory / "project.json"

    def test_round_trip_multiple_sequences_and_relative_assets(self):
        project = app.new_project(self.params)
        first = project.sequences[0]
        first.shots = [app.Shot(replace(self.params, angle=45)),
                       app.Shot(replace(self.params, anim_cycles=2, anim_fps=24))]
        second = app.Sequence("Other", [app.Shot(replace(self.params, width=32))])
        project.sequences.append(second)
        project.active_id = second.id
        app.save_project_file(project, self.path)
        values = json.loads(self.path.read_text())
        self.assertEqual(values["editor"]["background_images"], ["road.png"])
        loaded = app.load_project_file(self.path)
        self.assertEqual(loaded, project)
        self.assertEqual(loaded.sequences[0].shots[0].params.angle, 45)
        self.assertEqual(loaded.editor.angle, 0)
        self.assertIsNot(loaded.editor, loaded.sequences[0].shots[0].params)

    def test_legacy_sequences_open_without_rewriting_and_round_trip(self):
        project = app.new_project(self.params)
        project.sequences[0].shots = [
            app.Shot(replace(self.params, angle=45)),
            app.Shot(replace(self.params, anim_cycles=2)),
        ]
        project.sequences.append(app.Sequence(
            "Other", [app.Shot(replace(self.params, width=32))]))
        project.active_id = project.sequences[1].id
        values = app.project_values(project, self.directory)
        setups = [values["editor"]] + [
            shot["params"] for sequence in values["sequences"] for shot in sequence["shots"]]
        for setup in setups:
            del setup["shadows_enabled"]
            del setup["shadow_style"]
        self.path.write_text(json.dumps(values), encoding="utf-8")
        original = self.path.read_bytes()

        project.editor = replace(project.editor, shadow_style="cast")
        for sequence in project.sequences:
            for shot in sequence.shots:
                shot.params = replace(shot.params, shadow_style="cast")
        loaded = app.load_project_file(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(loaded, project)
        app.save_project_file(loaded, self.path)
        self.assertEqual(app.load_project_file(self.path), project)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        for setup in [saved["editor"]] + [
                shot["params"] for sequence in saved["sequences"] for shot in sequence["shots"]]:
            self.assertTrue(setup["shadows_enabled"])
            self.assertEqual(setup["shadow_style"], "cast")

    def test_shadow_settings_round_trip_and_reject_invalid_style(self):
        project = app.new_project(replace(self.params, shadows_enabled=False,
                                          shadow_style="cast"))
        project.sequences[0].shots.append(app.Shot(replace(project.editor)))
        app.save_project_file(project, self.path)
        self.assertEqual(app.load_project_file(self.path), project)
        values = json.loads(self.path.read_text(encoding="utf-8"))
        values["sequences"][0]["shots"][0]["params"]["shadow_style"] = "invalid"
        self.path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shadow style"):
            app.load_project_file(self.path)

    def test_failed_atomic_save_preserves_previous_project(self):
        project = app.new_project(self.params)
        app.save_project_file(project, self.path)
        previous = self.path.read_bytes()
        project.editor.angle = 40
        with patch.object(app.os, "replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                app.save_project_file(project, self.path)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_rejects_malformed_projects_and_missing_backgrounds(self):
        project = app.new_project(self.params)
        project.sequences[0].shots.append(app.Shot(replace(self.params)))
        values = app.project_values(project, self.directory)
        bad_values = [
            {**values, "version": 2},
            {**values, "version": True},
            {**values, "sequences": []},
            {**values, "active_sequence": "missing"},
            {**values, "editor": {**values["editor"], "width": 0}},
            {**values, "editor": {**values["editor"], "zoom": float("nan")}},
            {**values, "editor": {**values["editor"], "background_images": [1]}},
            {**values, "editor": {key: value for key, value in values["editor"].items()
                                  if key != "angle"}},
            {**values, "editor": {}},
        ]
        duplicate = json.loads(json.dumps(values))
        duplicate["sequences"].append(duplicate["sequences"][0])
        bad_values.append(duplicate)
        for bad in bad_values:
            with self.subTest(bad=bad):
                self.path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    app.load_project_file(self.path)
        self.path.write_text(json.dumps(values))
        self.image.unlink()
        with self.assertRaisesRegex(ValueError, "Cannot read background"):
            app.load_project_file(self.path)

    def test_names_are_safe_filename_prefixes(self):
        for name in ("", " ", "../bad", "one/two", "bad*", "con", "LPT1", "last.",
                     " leading", "trailing ", "new\nline"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                app.validate_sequence_name(name)
        self.assertEqual(app.validate_sequence_name("Sequence 2"), "Sequence 2")


class SpriteTests(unittest.TestCase):
    def test_lazy_frames_match_eager_recoloring_and_cache_only_accessed_frames(self):
        rng = np.random.default_rng(42)
        layers = {
            layer: [(rng.random((3, 4), dtype=np.float32).astype(np.float16),
                     rng.integers(0, 256, (3, 4), dtype=np.uint8))
                    for _ in range(app.crowd.FRAME_COUNT)]
            for layer in app.crowd.BODY_LAYERS + app.crowd.HAIR_LAYERS
        }
        data = {"size": (4, 3), "anchor": (1, 3), "layers": layers}
        colors = {"legs": (30, 70, 110), "arms": (170, 20, 200),
                  "head": (200, 150, 100), "hair": (40, 30, 20)}
        original_composite = Image.alpha_composite
        for female, mirror in ((False, False), (True, True)):
            with self.subTest(female=female, mirror=mirror), \
                    patch.object(app.crowd, "load_assets", return_value={"away": data}), \
                    patch.object(Image, "alpha_composite", wraps=original_composite) as composite:
                frames, anchor = app.crowd._build_frames("away", colors, female, mirror)
                composite.assert_not_called()
                first = frames[0]
                self.assertIs(frames[0], first)
                self.assertEqual(composite.call_count, 4)
                self.assertEqual(anchor, (3 if mirror else 1, 3))
                for index in range(app.crowd.FRAME_COUNT):
                    expected = Image.new("RGBA", (4, 3))
                    for layer, color in (("legs", colors["legs"]), ("arms", colors["arms"]),
                                         ("head", colors["head"]),
                                         ("hair_female" if female else "hair_male", colors["hair"])):
                        norm, alpha = layers[layer][index]
                        rgb = np.clip(norm[..., None].astype(np.float32) *
                                      np.asarray(color, dtype=np.float32), 0, 255).astype(np.uint8)
                        expected = original_composite(expected, Image.fromarray(np.dstack([rgb, alpha])))
                    if mirror:
                        expected = expected.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    self.assertEqual(frames[index].tobytes(), expected.tobytes())
                self.assertEqual(composite.call_count, 4 * app.crowd.FRAME_COUNT)


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.params = app.Params(width=128, height=128, zoom=12, tilt=0)
        self.cam = app.Camera(self.params, 128, 128)
        self.sprite = Image.new("RGBA", (100, 100))
        draw = ImageDraw.Draw(self.sprite)
        draw.ellipse((35, 5, 65, 30), fill=(200, 80, 40, 220))
        draw.rectangle((30, 30, 70, 65), fill=(200, 80, 40, 180))
        draw.rectangle((30, 65, 42, 99), fill=(200, 80, 40, 220))
        draw.rectangle((58, 65, 70, 99), fill=(200, 80, 40, 220))
        self.person = SimpleNamespace(
            height=2, frames=[self.sprite] * app.crowd.FRAME_COUNT,
            anchor=(50, 100), phase=0, rate=0,
            ground_pose=lambda t, obstacle: (0, 0))
        self.dog = SimpleNamespace(
            x=0, y=0, height=2, WORK_HEIGHT=100,
            obstacle=lambda: None,
            sprite=lambda cam, people, t: (self.sprite, (50, 100)))

    def image(self, people=(), dog=None, **options):
        image = Image.new("RGB", (128, 128), "white")
        return app.crowd.draw_frame(image, self.cam, people, 0, dog, **options)

    def test_both_characters_have_shadows_and_disabled_is_unchanged(self):
        for people, dog in (([self.person], None), ([], self.dog)):
            with self.subTest(dog=dog is not None):
                self.cam = app.Camera(replace(self.params, tilt=45), 128, 128)
                disabled = self.image(people, dog)
                self.assertEqual(disabled.tobytes(),
                                 self.image(people, dog, shadows=False).tobytes())
                for style in ("contact", "cast"):
                    enabled = self.image(people, dog, shadows=True, shadow_style=style)
                    self.assertIsNotNone(ImageChops.difference(disabled, enabled).getbbox())

    def test_shadow_follows_sprite_alpha_and_animation(self):
        blank = Image.new("RGBA", self.sprite.size)
        self.person.frames = [blank] * app.crowd.FRAME_COUNT
        empty = self.image([self.person], shadows=True, shadow_style="cast")
        self.assertIsNone(ImageChops.difference(
            empty, Image.new("RGB", empty.size, "white")).getbbox())
        self.person.frames[1] = self.sprite
        self.person.rate = 1
        image = Image.new("RGB", (128, 128), "white")
        animated = app.crowd.draw_frame(image, self.cam, [self.person], 1,
                                        shadows=True, shadow_style="cast")
        without = Image.new("RGB", (128, 128), "white")
        app.crowd.draw_frame(without, self.cam, [self.person], 1)
        self.assertIsNotNone(ImageChops.difference(animated, without).getbbox())

    def test_contact_shadow_is_a_blob_independent_of_sprite_shape(self):
        cam = app.Camera(self.params, 512, 512)
        other = Image.new("RGBA", (100, 100))
        ImageDraw.Draw(other).rectangle((5, 5, 20, 50), fill=(0, 0, 0, 255))
        masks = []
        for sprite in (self.sprite, other):
            mask = Image.new("L", (512, 512))
            app.crowd._draw_shadow(mask, cam, sprite, (50, 100), 100,
                                   0, 0, 2, 0, "contact", "test", 0)
            masks.append(mask)
        self.assertEqual(masks[0].tobytes(), masks[1].tobytes())
        bounds = masks[0].getbbox()
        self.assertIsNotNone(bounds)
        self.assertGreater(masks[0].getpixel((256, 256)), 0)
        filled = sum(value > 0 for value in masks[0].crop(bounds).getdata())
        self.assertGreater(filled, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 0.45)
        empty = Image.new("L", (512, 512))
        app.crowd._draw_shadow(empty, cam, Image.new("RGBA", (100, 100)),
                               (50, 100), 100, 0, 0, 2, 0, "contact", "test", 0)
        self.assertIsNone(empty.getbbox())

    def test_cast_shadow_projects_head_away_from_feet_and_preserves_alpha(self):
        def shadow(box, alpha):
            sprite = Image.new("RGBA", (100, 100))
            ImageDraw.Draw(sprite).rectangle(box, fill=(0, 0, 0, alpha))
            mask = Image.new("L", (128, 128))
            app.crowd._draw_shadow(mask, self.cam, sprite, (50, 100), 100,
                                   0, 0, 2, 0, "cast", "test", 0)
            return mask
        head = shadow((45, 5, 55, 15), 255)
        feet = shadow((45, 85, 55, 95), 255)
        head_box, feet_box = head.getbbox(), feet.getbbox()
        self.assertIsNotNone(head_box)
        self.assertIsNotNone(feet_box)
        self.assertGreater(head_box[0], feet_box[0] + 5)
        self.assertLess(head_box[1], feet_box[1] - 5)
        lighter = shadow((45, 5, 55, 15), 64)
        self.assertLess(lighter.getextrema()[1], head.getextrema()[1] / 2)

    def test_shadow_preserves_sprite_left_right_orientation(self):
        for angle in (0, 45, 90, 180, 270):
            cam = app.Camera(replace(self.params, angle=angle, tilt=45), 512, 512)
            along = 2 * 0.85 * (100 - 45) / 100 / 2**0.5
            center_x = cam.project(cam.to_cam(along, along))[0]
            for box, sign in (((70, 40, 90, 50), 1), ((10, 40, 30, 50), -1)):
                with self.subTest(angle=angle, sign=sign):
                    sprite = Image.new("RGBA", (100, 100))
                    ImageDraw.Draw(sprite).rectangle(box, fill=(0, 0, 0, 255))
                    mask = Image.new("L", (512, 512))
                    app.crowd._draw_shadow(mask, cam, sprite, (50, 100), 100,
                                           0, 0, 2, 0, "cast", "test", 0)
                    bounds = mask.getbbox()
                    self.assertIsNotNone(bounds)
                    self.assertGreater(sign * ((bounds[0] + bounds[2]) / 2 - center_x), 1)

    def test_styles_and_drawing_hold_are_deterministic(self):
        def frame(number, style):
            image = Image.new("RGB", (128, 128), "white")
            return app.crowd.draw_frame(image, self.cam, [self.person], number,
                                        shadows=True, shadow_style=style, shadow_hold=2)
        contact = frame(0, "contact")
        self.assertEqual(contact.tobytes(), frame(1, "contact").tobytes())
        self.assertEqual(contact.tobytes(), frame(0, "contact").tobytes())
        self.assertNotEqual(contact.tobytes(), frame(2, "contact").tobytes())
        self.assertNotEqual(contact.tobytes(), frame(0, "cast").tobytes())

    def test_cast_shadow_visible_when_sprite_is_offscreen(self):
        self.person.anchor = (100, 100)
        self.person.ground_pose = lambda t, obstacle: (-6.5, 0)
        without = self.image([self.person])
        with_shadow = self.image([self.person], shadows=True, shadow_style="cast")
        self.assertIsNotNone(ImageChops.difference(without, with_shadow).getbbox())

    def test_extreme_camera_and_near_plane(self):
        for tilt, x, y in ((0, 0, 0), (70, 0, 0), (70, 0, -30)):
            with self.subTest(tilt=tilt, x=x, y=y):
                self.cam = app.Camera(replace(self.params, tilt=tilt), 128, 128)
                self.person.ground_pose = lambda t, obstacle: (x, y)
                image = self.image([self.person], shadows=True, shadow_style="cast")
                self.assertEqual(image.size, (128, 128))


class SequenceUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.paths = []
        for color in ("red", "blue"):
            path = self.directory / f"{color}.png"
            Image.new("RGB", (20, 20), color).save(path)
            self.paths.append(str(path))
        self.params = app.Params(width=16, height=16, crowd_enabled=False,
                                 dog_enabled=False, stripe_wobble=0, anim_cycles=1,
                                 background_perspective=False,
                                 background_images=tuple(self.paths), dog_hold=2)
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.no_stripes = patch.object(app, "_stripe_runs", return_value=[])
        self.no_stripes.start()
        self.addCleanup(self.no_stripes.stop)
        with patch.object(app, "load_defaults", return_value=replace(self.params)), \
                patch.object(app, "_BackgroundWorker", _ThreadWorker):
            self.ui = app.App(self.root)
        self.addCleanup(self._cancel_jobs)
        self.addCleanup(app._road_texture.cache_clear)
        self.addCleanup(app._tiled_road_texture.cache_clear)
        self.addCleanup(self._shutdown_worker)
        self._cancel_jobs()

    def _shutdown_worker(self):
        self.ui._closed = True
        if self.ui._worker_cancel is not None:
            self.ui._worker_cancel.set()
        self.ui._executor.shutdown(wait=True, cancel_futures=True)

    def _cancel_jobs(self):
        for name in ("_job", "_anim_job", "_worker_job", "_cards_job"):
            job = getattr(self.ui, name)
            if job is not None:
                self.root.after_cancel(job)
                setattr(self.ui, name, None)

    def _start_render(self):
        with patch.object(app.filedialog, "askdirectory", return_value=str(self.directory)):
            self.ui.render_sequence()

    def _drain_render(self):
        self._wait_until(lambda: not self.ui._rendering)

    def _wait_until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate():
            self.assertLess(time.monotonic(), deadline, "Background operation timed out")
            self.root.update()
            time.sleep(0.002)

    def _assert_heartbeat_until_idle(self, release):
        beats = []
        job = None

        def beat():
            nonlocal job
            beats.append(time.monotonic())
            job = self.root.after(10, beat)

        beat()
        release_job = self.root.after(200, release.set)
        try:
            self._drain_render()
        finally:
            release.set()
            self.root.after_cancel(release_job)
            self.root.after_cancel(job)
        self.assertGreaterEqual(len(beats), 8)
        self.assertLess(max(b - a for a, b in zip(beats, beats[1:])), 0.250)

    def test_add_snapshots_frame_zero_and_explicit_update_after_reorder(self):
        self.ui.vars["animate_preview"].set(True)
        self.ui._anim_t = 3
        self.ui.add_shot()
        first = self.ui.sequence.shots[0]
        self._wait_until(lambda: first.id in self.ui._thumbnails)
        thumbnail = ImageTk.getimage(self.ui._thumbnails[first.id])
        self.assertEqual(thumbnail.getpixel((40, 40))[:3], (255, 0, 0))
        self.ui.vars["angle"].set(45)
        self.ui.add_shot()
        self.assertEqual(first.params.angle, 0)
        self.assertIsNot(first.params, self.ui.p)
        self.ui.edit_shot(first.id)
        self.assertFalse(self.ui.vars["animate_preview"].get())
        self.assertEqual(self.ui._preview_frame, 0)
        self.ui.vars["angle"].set(90)
        self.assertEqual(first.params.angle, 0)
        self.ui.move_shot(first.id, 1)
        self.assertEqual(self.ui.sequence.shots[1].id, first.id)
        self.assertTrue(self.ui.update_shot())
        self.assertEqual(first.params.angle, 90)
        self.assertEqual(self.ui.sequence.shots[0].params.angle, 45)

    def test_drag_reorders_cards_and_removal_clears_edit_target(self):
        for angle in (0, 45, 90):
            self.ui.vars["angle"].set(angle)
            self.ui.add_shot()
        first = self.ui.sequence.shots[0]
        self.ui.edit_shot(first.id)
        self.root.update_idletasks()
        self.ui._start_drag(first.id)
        last_card = self.ui._card_widgets[-1]
        self.ui._drag_motion(SimpleNamespace(
            x_root=last_card.winfo_rootx() + last_card.winfo_width()))
        self.assertEqual(self.ui._drag_target, 3)
        self.ui._drop_shot(SimpleNamespace())
        self.assertEqual([s.params.angle for s in self.ui.sequence.shots], [45, 90, 0])
        self.assertEqual(self.ui._edit_id, first.id)
        with patch.object(app.messagebox, "askyesno", return_value=True):
            self.ui.remove_shot(first.id)
        self.assertIsNone(self.ui._edit_id)
        self.assertNotIn(first.id, self.ui._thumbnails)

    def test_pending_shot_edits_can_cancel_switch_or_be_discarded(self):
        self.ui.add_shot()
        self.ui.add_shot()
        first, second = self.ui.sequence.shots
        self.ui.edit_shot(first.id)
        self.ui.vars["angle"].set(60)
        with patch.object(app.messagebox, "askyesnocancel", return_value=None):
            self.ui.edit_shot(second.id)
        self.assertEqual(self.ui._edit_id, first.id)
        self.assertEqual(self.ui.p.angle, 60)
        with patch.object(app.messagebox, "askyesnocancel", return_value=False):
            self.ui.edit_shot(second.id)
        self.assertEqual(self.ui._edit_id, second.id)
        self.assertEqual(first.params.angle, 0)
        self.assertEqual(self.ui.p.angle, 0)

    def test_save_keeps_uncommitted_editor_separate_and_open_restores(self):
        self.ui.add_shot()
        shot = self.ui.sequence.shots[0]
        self.ui.edit_shot(shot.id)
        self.ui.vars["angle"].set(60)
        path = self.directory / "saved.json"
        with patch.object(app.filedialog, "asksaveasfilename", return_value=str(path)):
            self.assertTrue(self.ui.save_project())
        loaded = app.load_project_file(path)
        self.assertEqual(loaded.editor.angle, 60)
        self.assertEqual(loaded.sequences[0].shots[0].params.angle, 0)
        self.assertTrue(self.ui._allow_replace_project())
        with patch.object(app.filedialog, "askopenfilename", return_value=str(path)):
            self.ui.open_project()
            self._drain_render()
        self.assertEqual(self.ui.p.angle, 60)
        self.assertIsNone(self.ui._edit_id)
        self.assertEqual(self.ui.sequence.shots[0].params.angle, 0)

    def test_failed_open_and_cancelled_new_preserve_project(self):
        self.ui.add_shot()
        project = self.ui.project
        invalid = self.directory / "bad.json"
        invalid.write_text("{}")
        with patch.object(app.filedialog, "askopenfilename", return_value=str(invalid)), \
                patch.object(app.messagebox, "showerror") as error:
            self.ui.open_project()
            self._drain_render()
        error.assert_called_once()
        self.assertIs(self.ui.project, project)
        with patch.object(app.messagebox, "askyesnocancel", return_value=None):
            self.ui.new_project()
        self.assertIs(self.ui.project, project)

    def test_named_sequences_reject_duplicates_and_save_active_selection(self):
        with patch.object(app.simpledialog, "askstring", return_value="Other"):
            self.ui.add_sequence()
        self.ui.add_shot()
        other = self.ui.sequence
        with patch.object(app.simpledialog, "askstring", return_value="sequence1"), \
                patch.object(app.messagebox, "showerror") as error:
            self.ui.rename_sequence()
        error.assert_called_once()
        self.assertEqual(other.name, "Other")
        self.ui.sequence_choice.current(0)
        self.ui._select_sequence()
        self.assertEqual(self.ui.sequence.name, "Sequence1")
        self.assertEqual(other.shots[0].params, self.ui.p)
        self.assertEqual(app.project_values(self.ui.project, self.directory)["active_sequence"],
                         self.ui.sequence.id)

    def test_export_exact_names_order_sizes_and_frame_reset(self):
        self.ui.add_shot()
        self.ui.p.background_images = (self.paths[1], self.paths[0])
        self.ui.vars["width"].set(32)
        self.ui.vars["anim_cycles"].set(2)
        self.ui.vars["anim_fps"].set(24)
        self.ui.add_shot()
        self.ui.move_shot(self.ui.sequence.shots[1].id, -1)
        self._start_render()
        self.assertTrue(self.ui._rendering)
        self._drain_render()
        expected = {f"Sequence1_shot{shot}.{frame:04d}.png"
                    for shot, total in ((1, 24), (2, 12)) for frame in range(1, total + 1)}
        self.assertEqual({p.name for p in self.directory.glob("Sequence*.png")}, expected)
        for filename, size, color in (
                ("Sequence1_shot1.0001.png", (32, 16), (0, 0, 255)),
                ("Sequence1_shot1.0003.png", (32, 16), (255, 0, 0)),
                ("Sequence1_shot2.0001.png", (16, 16), (255, 0, 0))):
            with Image.open(self.directory / filename) as image:
                self.assertEqual(image.size, size)
                self.assertEqual(image.getpixel((8, 8)), color)
        self.assertIn("Complete: 36/36", self.ui.status.cget("text"))
        self.assertEqual(str(self.ui.sequence_choice.cget("state")), "readonly")

    def test_static_shot_and_conflicting_destination(self):
        self.ui.p.background_images = (self.paths[0],)
        self.ui.add_shot()
        outputs = list(app.sequence_outputs(self.ui.sequence, self.directory))
        self.assertEqual(len(outputs), 12)
        self.assertEqual(outputs[0][1].tobytes(), outputs[-1][1].tobytes())
        existing = self.directory / "Sequence1_shot99.9999.png"
        existing.write_bytes(b"do not replace")
        with patch.object(app.messagebox, "showerror") as error:
            self._start_render()
            self._drain_render()
        error.assert_called_once()
        self.assertFalse(self.ui._rendering)
        self.assertEqual(existing.read_bytes(), b"do not replace")
        self.assertEqual(list(self.directory.glob("Sequence1_shot1.*.png")), [])

    def test_shot_renderer_matches_existing_crowd_and_dog_composition(self):
        self.no_stripes.stop()
        p = replace(self.params, width=64, height=64, crowd_enabled=True,
                    crowd_count=2, dog_enabled=True, stripe_wobble=0.25)
        renderer = app.ShotRenderer(p)
        self.assertEqual(len(renderer.people), 2)
        for style, number in (("contact", 0), ("contact", 5), ("cast", 5)):
            with self.subTest(style=style, frame=number):
                p.shadow_style = style
                renderer = app.ShotRenderer(p)
                bg = app.render(p, frame=number)
                expected = self.ui._compose_crowd(
                    bg, p, app.Camera(p, p.width, p.height), number)
                actual = renderer.frame(number)
                self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertNotEqual(actual.tobytes(), bg.tobytes())
                if number == 0:
                    sequence = app.Sequence("Test", [app.Shot(replace(p))])
                    exported = next(app.sequence_outputs(sequence, self.directory))[1]
                    self.assertEqual(exported.tobytes(), actual.tobytes())

    def test_shadow_controls_follow_shot_edits(self):
        self.ui.vars["shadows_enabled"].set(False)
        self.ui.vars["shadow_style"].set("cast")
        self.ui.add_shot()
        shot = self.ui.sequence.shots[0]
        self.assertFalse(shot.params.shadows_enabled)
        self.assertEqual(shot.params.shadow_style, "cast")
        self.ui.vars["shadows_enabled"].set(True)
        self.ui.vars["shadow_style"].set("contact")
        self.ui.edit_shot(shot.id)
        self.assertFalse(self.ui.vars["shadows_enabled"].get())
        self.assertEqual(self.ui.vars["shadow_style"].get(), "cast")

    def test_cancellation_retains_frames_and_restores_controls(self):
        self.ui.add_shot()
        saved = threading.Event()
        release = threading.Event()
        original_save = Image.Image.save

        def delayed_save(image, *args, **kwargs):
            original_save(image, *args, **kwargs)
            saved.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release the PNG write")

        with patch.object(Image.Image, "save", delayed_save):
            try:
                self._start_render()
                self._wait_until(saved.is_set)
                self.ui.cancel_sequence_render()
                self.assertTrue(self.ui._rendering)
                self.assertEqual(str(self.ui.sequence_choice.cget("state")), "disabled")
            finally:
                release.set()
            self._drain_render()
        self.assertFalse(self.ui._rendering)
        self.assertEqual(len(list(self.directory.glob("Sequence*.png"))), 1)
        self.assertIn("Cancelled: 1/12", self.ui.status.cget("text"))
        self.assertEqual(str(self.ui.cancel_render_button.cget("state")), "disabled")

    def test_scheduled_render_completes_through_tk_event_loop(self):
        self.ui.add_shot()
        self._start_render()

        def finish_when_idle():
            if self.ui._rendering:
                self.root.after(5, finish_when_idle)
            else:
                self.root.quit()

        watchdog = self.root.after(5000, self.root.quit)
        self.root.after(5, finish_when_idle)
        self.root.mainloop()
        self.root.after_cancel(watchdog)
        self.assertFalse(self.ui._rendering)
        self.assertEqual(len(list(self.directory.glob("Sequence*.png"))), 12)
        self.assertIn("Complete: 12/12", self.ui.status.cget("text"))

    def test_write_failure_reports_partial_output_and_no_success(self):
        self.ui.add_shot()
        with patch.object(Image.Image, "save", side_effect=OSError("disk full")), \
                patch.object(app.messagebox, "showerror") as error:
            self._start_render()
            self._drain_render()
        error.assert_called_once()
        self.assertIn("complete frames retained", error.call_args.args[1])
        self.assertFalse(self.ui._rendering)
        self.assertIn("Failed: 0/12", self.ui.status.cget("text"))

    def test_project_load_is_responsive_and_transactional(self):
        project = self.ui.project
        loaded = app.new_project(replace(self.params, angle=45))
        loaded.sequences[0].shots = [app.Shot(replace(self.params)) for _ in range(20)]
        started, release = threading.Event(), threading.Event()
        initial_cards = []
        install = self.ui._install_project

        def installed(*args):
            install(*args)
            initial_cards.append((len(self.ui._card_widgets), len(self.ui._thumbnails)))

        def load(path):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release project loading")
            return loaded

        with patch.object(app, "load_project_file", side_effect=load), \
                patch.object(self.ui, "_install_project", side_effect=installed), \
                patch.object(app.filedialog, "askopenfilename", return_value="example.json"):
            try:
                self.ui.open_project()
                self._wait_until(started.is_set)
                self.assertIs(self.ui.project, project)
                self.assertEqual(str(self.ui.sequence_choice.cget("state")), "disabled")
                self._assert_heartbeat_until_idle(release)
            finally:
                release.set()
        self.assertIs(self.ui.project, loaded)
        self.assertEqual(self.ui.p.angle, 45)
        self.assertEqual(initial_cards, [(8, 0)])
        self._wait_until(lambda: len(self.ui._thumbnails) == 20)
        self.assertEqual(len(self.ui._card_widgets), 20)

    def test_cancelled_load_keeps_old_project_until_worker_acknowledges(self):
        original = self.ui.project
        started, release = threading.Event(), threading.Event()

        def load(path):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release project loading")
            return app.new_project(self.params)

        with patch.object(app, "load_project_file", side_effect=load), \
                patch.object(app.filedialog, "askopenfilename", return_value="example.json"):
            try:
                self.ui.open_project()
                self._wait_until(started.is_set)
                self.ui.cancel_sequence_render()
                self.assertTrue(self.ui._rendering)
                self.assertIs(self.ui.project, original)
            finally:
                release.set()
            self._drain_render()
        self.assertIs(self.ui.project, original)
        self.assertEqual(str(self.ui.sequence_choice.cget("state")), "readonly")

    def test_superseded_thumbnail_and_preview_results_are_not_applied(self):
        started, release = threading.Event(), threading.Event()

        def thumbnail(p):
            if p.angle == 0:
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release thumbnail")
            return Image.new("RGB", (90, 90), "red" if p.angle == 0 else "blue")

        with patch.object(app, "_thumbnail_image", side_effect=thumbnail):
            try:
                self.ui.add_shot()
                shot = self.ui.sequence.shots[0]
                self._wait_until(started.is_set)
                self.ui.edit_shot(shot.id)
                self.ui.vars["angle"].set(45)
                self.ui.update_shot()
            finally:
                release.set()
            self._wait_until(lambda: shot.id in self.ui._thumbnails)
            self.assertEqual(ImageTk.getimage(self.ui._thumbnails[shot.id]).getpixel((0, 0))[:3],
                             (0, 0, 255))

        started.clear()
        release.clear()

        def preview(p, width, height, frame):
            if p.angle == 45:
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release preview")
            image = Image.new("RGB", (width, height), "red" if p.angle == 45 else "blue")
            return image, image.copy()

        with patch.object(app, "_preview_images", side_effect=preview):
            try:
                self.ui.update_preview()
                self._wait_until(started.is_set)
                self.ui.vars["angle"].set(90)
            finally:
                release.set()
            self._wait_until(lambda: self.ui._preview_bg is not None
                             and self.ui._preview_bg.getpixel((0, 0)) == (0, 0, 255))

    def test_sequence_and_animation_exports_remain_responsive_during_writes(self):
        self.ui.add_shot()
        original_save = Image.Image.save
        for animation in (False, True):
            started, release = threading.Event(), threading.Event()

            def save(image, *args, **kwargs):
                if not release.is_set():
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("Test did not release PNG write")
                return original_save(image, *args, **kwargs)

            with self.subTest(animation=animation), patch.object(Image.Image, "save", save), \
                    patch.object(app.filedialog, "asksaveasfilename",
                                 return_value=str(self.directory / "animation.gif")):
                try:
                    if animation:
                        self.ui.save_animation()
                    else:
                        self._start_render()
                    self._wait_until(started.is_set)
                    self._assert_heartbeat_until_idle(release)
                finally:
                    release.set()
                self.assertIn("Complete: 12/12", self.ui.status.cget("text"))

    def test_animation_cancellation_and_gif_failure_restore_controls(self):
        output = self.directory / "animation.gif"
        original_save = Image.Image.save
        saved, release = threading.Event(), threading.Event()

        def delayed_save(image, *args, **kwargs):
            original_save(image, *args, **kwargs)
            saved.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release PNG write")

        with patch.object(app.filedialog, "asksaveasfilename", return_value=str(output)):
            with patch.object(Image.Image, "save", delayed_save):
                try:
                    self.ui.save_animation()
                    self._wait_until(saved.is_set)
                    self.ui.cancel_sequence_render()
                    self.assertTrue(self.ui._rendering)
                finally:
                    release.set()
                self._drain_render()
            self.assertFalse(output.exists())
            self.assertEqual(len(list(self.directory.glob("animation_*.png"))), 1)
            self.assertIn("Cancelled: 1/12", self.ui.status.cget("text"))

            def failing_gif(image, path, *args, **kwargs):
                if str(path).endswith(".gif"):
                    raise OSError("GIF encoding failed")
                return original_save(image, path, *args, **kwargs)

            with patch.object(Image.Image, "save", failing_gif), \
                    patch.object(app.messagebox, "showerror") as error:
                self.ui.save_animation()
                self._drain_render()
            error.assert_called_once()
            self.assertIn("GIF encoding failed", error.call_args.args[1])
            self.assertIn("Failed: 12/12", self.ui.status.cget("text"))
            self.assertEqual(str(self.ui.sequence_choice.cget("state")), "readonly")

    def test_animation_preserves_gif_timing_loop_and_png_frames(self):
        self.no_stripes.stop()
        for crowd_on in (False, True):
            with self.subTest(crowd=crowd_on):
                self.ui.vars["crowd_enabled"].set(crowd_on)
                self.ui.vars["crowd_count"].set(1)
                p = replace(self.ui.read_params())
                output = self.directory / f"animation_{crowd_on}.gif"
                with patch.object(app.filedialog, "asksaveasfilename", return_value=str(output)):
                    self.ui.save_animation()
                    self._drain_render()
                with Image.open(output) as gif:
                    self.assertEqual(gif.size, (16, 16))
                    if crowd_on:
                        self.assertNotIn("loop", gif.info)
                    else:
                        self.assertEqual(gif.info["loop"], 0)
                        self.assertEqual(gif.n_frames, 6)
                        for index in range(gif.n_frames):
                            gif.seek(index)
                            self.assertEqual(gif.info["duration"], 160)
                renderer = app.ShotRenderer(p)
                for index in range(12):
                    with Image.open(output.with_name(f"{output.stem}_{index + 1:04d}.png")) as png:
                        self.assertEqual(png.tobytes(), renderer.frame(index).tobytes())


class ProcessWorkflowTests(unittest.TestCase):
    def test_real_worker_load_preview_exports_errors_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            backgrounds = []
            for color in ("red", "blue"):
                path = directory / f"{color}.png"
                Image.new("RGB", (20, 20), color).save(path)
                backgrounds.append(str(path))
            p = app.Params(width=32, height=24, anim_cycles=1, crowd_enabled=False,
                           dog_enabled=False, background_images=tuple(backgrounds),
                           background_perspective=False, stripe_wobble=0)
            project = app.new_project(p)
            project.sequences[0].shots = [app.Shot(replace(p)), app.Shot(replace(p, width=48))]
            path = directory / "project.json"
            app.save_project_file(project, path)
            root = tk.Tk()
            root.withdraw()
            with patch.object(app, "load_defaults", return_value=replace(p)):
                ui = app.App(root)
            if ui._job is not None:
                root.after_cancel(ui._job)
                ui._job = None

            def wait(predicate):
                expired = []

                def poll():
                    if predicate():
                        root.quit()
                    else:
                        root.after(10, poll)

                def timeout():
                    expired.append(True)
                    root.quit()

                watchdog = root.after(15000, timeout)
                root.after(10, poll)
                root.mainloop()
                root.after_cancel(watchdog)
                self.assertFalse(expired, "Worker process timed out")

            try:
                with patch.object(app.messagebox, "showerror") as error:
                    with patch.object(app.filedialog, "askopenfilename", return_value=str(path)):
                        ui.open_project()
                        wait(lambda: not ui._rendering)
                    self.assertEqual(ui.project, project)
                    wait(lambda: len(ui._thumbnails) == 2 and ui._future is None)

                    with patch.object(app, "render", side_effect=AssertionError("UI thread rendered")):
                        ui.update_preview()
                        wait(lambda: ui._preview_bg is not None and ui._future is None
                             and ui._pending_preview is None)
                        ui.vars["animate_preview"].set(True)
                        ui._anim_tick()
                        wait(lambda: ui._preview_frame == 1)
                        ui.vars["animate_preview"].set(False)
                        ui._toggle_anim_preview_stop()

                    with patch.object(app.filedialog, "askdirectory", return_value=str(directory)):
                        ui.render_sequence()
                        wait(lambda: not ui._rendering)
                    self.assertEqual(ui._render_written, 24)
                    self.assertIn("Complete: 24/24", ui.status.cget("text"))
                    self.assertEqual(len(list(directory.glob("Sequence1_shot*.png"))), 24)
                    with Image.open(directory / "Sequence1_shot2.0003.png") as png:
                        self.assertEqual(png.size, (48, 24))
                        self.assertEqual(png.tobytes(), app.ShotRenderer(replace(p, width=48)).frame(2).tobytes())

                    gif_path = directory / "animation.gif"
                    with patch.object(app.filedialog, "asksaveasfilename", return_value=str(gif_path)):
                        ui.save_animation()
                        wait(lambda: not ui._rendering)
                    self.assertEqual(ui._render_written, 12)
                    self.assertTrue(gif_path.exists())
                    error.assert_not_called()

                    cancel_directory = directory / "cancelled"
                    cancel_directory.mkdir()
                    with patch.object(app.filedialog, "askdirectory", return_value=str(cancel_directory)):
                        ui.render_sequence()
                        ui.cancel_sequence_render()
                        wait(lambda: not ui._rendering)
                    self.assertEqual(ui._render_written, 0)
                    self.assertEqual(list(cancel_directory.iterdir()), [])
                    self.assertIn("Cancelled: 0/24", ui.status.cget("text"))

                    original = ui.project
                    invalid = directory / "invalid.json"
                    invalid.write_text("{}")
                    with patch.object(app.filedialog, "askopenfilename", return_value=str(invalid)):
                        ui.open_project()
                        wait(lambda: not ui._rendering)
                    error.assert_called_once()
                    self.assertIs(ui.project, original)
                    self.assertEqual(str(ui.sequence_choice.cget("state")), "readonly")

                    Image.new("RGB", (20, 20), "green").save(backgrounds[0])
                    ui._install_project(app.load_project_file(path), path)
                    refreshed = directory / "refreshed"
                    refreshed.mkdir()
                    with patch.object(app.filedialog, "askdirectory", return_value=str(refreshed)):
                        ui.render_sequence()
                        wait(lambda: not ui._rendering)
                    with Image.open(refreshed / "Sequence1_shot1.0001.png") as png:
                        self.assertEqual(png.getpixel((0, 0)), (0, 128, 0))
            finally:
                root.destroy()
                ui._executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
