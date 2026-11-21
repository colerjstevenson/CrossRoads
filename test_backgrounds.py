import json
import math
import random
import tempfile
import time
import tkinter as tk
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import crosswalk_generator as app
from test_sequences import _ThreadWorker


class BackgroundTests(unittest.TestCase):
    def wait_until(self, root, predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            self.assertLess(time.monotonic(), deadline, "Background operation timed out")
            root.update()
            time.sleep(0.002)

    def cleanup_ui(self, ui):
        ui._closed = True
        if ui._worker_cancel is not None:
            ui._worker_cancel.set()
        ui._executor.shutdown(wait=True, cancel_futures=True)
        for name in ("_job", "_anim_job", "_worker_job", "_cards_job"):
            job = getattr(ui, name)
            if job is not None:
                ui.root.after_cancel(job)
                setattr(ui, name, None)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.paths = tuple(str(self.directory / f"{name}.png") for name in ("red", "blue"))
        for path, color in zip(self.paths, ("red", "blue")):
            Image.new("RGB", (20, 20), color).save(path)
        self.params = app.Params(width=16, height=16, crowd_enabled=False,
                                 dog_enabled=False, stripe_wobble=0,
                                 dog_hold=2, anim_cycles=1,
                                 background_images=self.paths)
        self.addCleanup(app._road_texture.cache_clear)
        self.addCleanup(app._tiled_road_texture.cache_clear)

    def test_render_cycles_on_hold_without_wobble(self):
        with patch.object(app, "_stripe_runs", return_value=[]):
            colors = [app.render(self.params, supersample=1, frame=frame).getpixel((8, 8))
                      for frame in (0, 1, 2, 3, 4)]
        self.assertEqual(colors, [(255, 0, 0), (255, 0, 0), (0, 0, 255),
                                  (0, 0, 255), (255, 0, 0)])

    def test_static_background_holds_first_image_and_stripes(self):
        for perspective in (False, True):
            with self.subTest(perspective=perspective):
                p = replace(self.params, width=64, height=64, stripe_wobble=1,
                            background_perspective=perspective)
                first = app.render(p, supersample=1, frame=0)
                animated = app.render(p, supersample=1, frame=2)
                self.assertNotEqual(first.tobytes(), animated.tobytes())
                p.background_animated = False
                p.stripes_animated = False
                renderer = app.ShotRenderer(p, supersample=1)
                for frame in (0, 1, 2, 4, 17):
                    self.assertEqual(app.render(p, supersample=1, frame=frame).tobytes(),
                                     first.tobytes())
                    self.assertEqual(renderer.frame(frame).tobytes(), first.tobytes())

    def test_stripes_and_background_animate_independently(self):
        for perspective in (False, True):
            with self.subTest(perspective=perspective):
                p = replace(self.params, width=64, height=64, stripe_wobble=1,
                            background_perspective=perspective)
                stripes_only = replace(p, background_animated=False)
                renderer = app.ShotRenderer(stripes_only, supersample=1)
                first = renderer.frame(0)
                later = renderer.frame(2)
                self.assertNotEqual(first.tobytes(), later.tobytes())
                with patch.object(app, "_stripe_runs", return_value=[]):
                    self.assertEqual(app.render(stripes_only, supersample=1, frame=0).tobytes(),
                                     app.render(stripes_only, supersample=1, frame=2).tobytes())

                road_only = replace(p, stripes_animated=False)
                renderer = app.ShotRenderer(road_only, supersample=1)
                first = renderer.frame(0)
                later = renderer.frame(2)
                self.assertNotEqual(first.tobytes(), later.tobytes())
                first_stripes = [pixel == (255, 255, 255) for pixel in first.getdata()]
                later_stripes = [pixel == (255, 255, 255) for pixel in later.getdata()]
                self.assertTrue(any(first_stripes))
                self.assertEqual(first_stripes, later_stripes)
                self.assertEqual(later.tobytes(), app.render(road_only, supersample=1, frame=2).tobytes())

    def test_texture_motion_can_be_disabled_independently(self):
        path = self.directory / "pattern.png"
        image = Image.new("RGB", (40, 40))
        image.putdata([(x * 6, y * 6, (x + y) * 3)
                       for y in range(40) for x in range(40)])
        image.save(path)
        p = replace(self.params, width=64, height=64, stripe_wobble=1,
                    background_images=(str(path),), stripes_animated=False)
        for perspective in (False, True):
            p.background_perspective = perspective
            with self.subTest(perspective=perspective), \
                 patch.object(app, "_stripe_runs", return_value=[]):
                first = app.render(p, supersample=1, frame=0)
                self.assertNotEqual(first.tobytes(), app.render(p, supersample=1, frame=2).tobytes())
                p.background_animated = False
                self.assertEqual(first.tobytes(), app.render(p, supersample=1, frame=2).tobytes())
                p.background_animated = True

    def test_tiled_background_uses_selected_image(self):
        self.params.background_zoom = 0.5
        with patch.object(app, "_stripe_runs", return_value=[]):
            red = app.render(self.params, supersample=1, frame=0).getpixel((8, 8))
            blue = app.render(self.params, supersample=1, frame=2).getpixel((8, 8))
        self.assertGreater(red[0], red[2])
        self.assertGreater(blue[2], blue[0])

    def test_tiled_background_does_not_add_grain(self):
        self.params.background_zoom = 0.5
        self.params.background_images = (self.paths[0],)
        with patch.object(app, "_stripe_runs", return_value=[]):
            image = app.render(self.params, supersample=1)
        self.assertEqual(image.getcolors(maxcolors=image.width * image.height),
                         [(image.width * image.height, (255, 0, 0))])

    def test_tiled_background_preserves_source_contrast(self):
        path = self.directory / "contrast.png"
        image = Image.new("RGB", (20, 20))
        image.putdata([(255, 255, 255) if (x + y) % 2 else (0, 0, 0)
                       for y in range(20) for x in range(20)])
        image.save(path)
        tiled = app._tiled_road_texture("#808080", str(path))
        self.assertEqual(tiled.getextrema()[0], (0, 255))

    def test_tiled_background_keeps_diagonal_direction(self):
        path = self.directory / "diagonal.png"
        image = Image.new("RGB", (40, 40))
        image.putdata([(3 * (x + y),) * 3
                       for y in range(40) for x in range(40)])
        image.save(path)
        tiled = app._tiled_road_texture("#808080", str(path))
        pitch = tiled.width // app.BACKGROUND_TILES
        for row in range(app.BACKGROUND_TILES):
            for col in range(app.BACKGROUND_TILES):
                with self.subTest(row=row, col=col):
                    x, y = col * pitch + pitch // 2, row * pitch + pitch // 2
                    center = tiled.getpixel((x, y))[0]
                    self.assertEqual(tiled.getpixel((x + 1, y))[0] - center, 3)
                    self.assertEqual(tiled.getpixel((x, y + 1))[0] - center, 3)

    def test_perspective_background_repeats_without_mirroring(self):
        texture = Image.new("RGB", (16, 12))
        texture.putdata([((x + y) % 16 * 16,) * 3
                         for y in range(12) for x in range(16)])
        params = app.Params(width=64, height=64, zoom=16, tilt=0,
                            stripe_wobble=0)
        rendered = app._draw_perspective_road(
            params, texture, texture.width, 64, 64,
            app.Camera(params, 64, 64), drawing=0, zoom=0.25)
        self.assertEqual(rendered.crop((0, 0, 48, 64)).tobytes(),
                         rendered.crop((16, 0, 64, 64)).tobytes())
        self.assertEqual(rendered.crop((0, 0, 64, 52)).tobytes(),
                         rendered.crop((0, 12, 64, 64)).tobytes())

    def test_perspective_background_recedes_with_camera(self):
        path = self.directory / "stripes.png"
        image = Image.new("RGB", (64, 64))
        image.putdata([
            (255, 255, 255) if (x // 4) % 2 else (0, 0, 0)
            for y in range(64) for x in range(64)
        ])
        image.save(path)
        params = app.Params(
            width=128, height=128, zoom=12, tilt=45, stripe_wobble=0,
            crowd_enabled=False, dog_enabled=False,
            background_images=(str(path),),
        )
        with patch.object(app, "_stripe_runs", return_value=[]):
            rendered = app.render(params, supersample=1)
        top_changes = sum(rendered.getpixel((x, 8)) != rendered.getpixel((x - 1, 8))
                          for x in range(1, rendered.width))
        bottom_changes = sum(
            rendered.getpixel((x, 120)) != rendered.getpixel((x - 1, 120))
            for x in range(1, rendered.width)
        )
        self.assertGreater(top_changes, bottom_changes)

    def test_block_sampling_matches_original_row_sampler_exactly(self):
        texture = Image.fromarray(
            np.random.default_rng(42).integers(0, 256, (13, 19, 3), dtype=np.uint8))

        def reference(p, width, height, cam, drawing, zoom):
            source = np.asarray(texture, dtype=np.uint8)
            source_height, texture_width = source.shape[:2]
            output = np.empty((height, width, 3), dtype=np.uint8)
            screen_x = (np.arange(width, dtype=np.float64) + 0.5 - width / 2) / cam.focal
            tilt = math.radians(p.tilt)
            sin_t, cos_t = math.sin(tilt), math.cos(tilt)
            shift_up = p.zoom * p.camera_shift / 100
            camera_y = -cam.dist * sin_t + shift_up * cos_t
            camera_z = cam.dist * cos_t + shift_up * sin_t
            scale = texture.width / (p.zoom * zoom)
            angle = math.radians((drawing * app.BACKGROUND_TURN_DEG) % 360) if p.stripe_wobble else 0
            ca, sa = math.cos(angle), math.sin(angle)
            shift = 0.12 * p.stripe_wobble * p.zoom * min(width, height) / width
            rng = random.Random(f"road:{drawing}")
            dx, dy = (rng.uniform(-shift, shift) for _ in range(2))
            for row in range(height):
                v = (cam.h / 2 - row - 0.5) / cam.focal
                denominator = cos_t - v * sin_t
                ray = camera_z / denominator if denominator > 1e-9 else cam.dist * 60
                ray = min(ray, cam.dist * 60)
                sx = ray * screen_x
                sy = camera_y + ray * (v * cos_t + sin_t)
                wx, wy = sx * cam.ca + sy * cam.sa, -sx * cam.sa + sy * cam.ca
                tx = texture_width / 2 + scale * (ca * wx + sa * wy + dx)
                ty = source_height / 2 - scale * (-sa * wx + ca * wy + dy)
                ix, iy = np.floor(tx).astype(np.int32), np.floor(ty).astype(np.int32)
                fx, fy = (tx - ix)[:, None].astype(np.float32), (ty - iy)[:, None].astype(np.float32)
                x0, x1 = ix % texture_width, (ix + 1) % texture_width
                y0, y1 = iy % source_height, (iy + 1) % source_height
                top = source[y0, x0].astype(np.float32) * (1 - fx) + source[y0, x1].astype(np.float32) * fx
                bottom = source[y1, x0].astype(np.float32) * (1 - fx) + source[y1, x1].astype(np.float32) * fx
                output[row] = np.clip(np.rint(top * (1 - fy) + bottom * fy), 0, 255).astype(np.uint8)
            return output.tobytes()

        for width, height in ((37, 23), (17, 49)):
            for tilt, angle, wobble, zoom, shift in (
                    (0, 0, 0, 1, 0), (45, 37, .35, .2, 0), (70, 180, 1, 2, 0),
                    (45, 37, .35, .2, 25), (70, 180, 0, 2, -75)):
                for drawing in (0, 1, 17):
                    with self.subTest(size=(width, height), tilt=tilt, drawing=drawing, shift=shift):
                        p = app.Params(tilt=tilt, angle=angle, stripe_wobble=wobble,
                                       camera_shift=shift)
                        cam = app.Camera(p, width, height)
                        actual = app._draw_perspective_road(
                            p, texture, texture.width, width, height, cam, drawing, zoom)
                        self.assertEqual(actual.tobytes(), reference(p, width, height, cam, drawing, zoom))

    def test_defaults_round_trip_and_reject_invalid_image_list(self):
        defaults = self.directory / "defaults.json"
        with patch.object(app, "DEFAULTS_PATH", defaults):
            app.save_defaults(self.params)
            self.assertEqual(app.load_defaults().background_images, self.paths)
            defaults.write_text(json.dumps({"background_images": ["ok", 1]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                app.load_defaults()
            defaults.write_text("{}", encoding="utf-8")
            self.assertEqual(app.load_defaults().background_images, ())
            self.assertEqual(app.load_defaults().shadow_style, "cast")
            self.assertTrue(app.load_defaults().background_animated)
            app.save_defaults(replace(self.params, background_animated=False))
            self.assertFalse(app.load_defaults().background_animated)
            self.assertTrue(app.load_defaults().stripes_animated)
            defaults.write_text('{"background_animated": false}', encoding="utf-8")
            self.assertFalse(app.load_defaults().stripes_animated)
            self.assertTrue(app.load_defaults().dog_blink)
            app.save_defaults(replace(self.params, dog_blink=False))
            self.assertFalse(app.load_defaults().dog_blink)
            defaults.write_text('{"shadows_enabled": false}', encoding="utf-8")
            self.assertFalse(app.load_defaults().shadows_enabled)
            self.assertEqual(app.load_defaults().shadow_style, "cast")
            defaults.write_text('{"shadow_style": "contact"}', encoding="utf-8")
            self.assertEqual(app.load_defaults().shadow_style, "contact")

    def test_preview_selection_and_animation_export(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        with patch.object(app, "load_defaults", return_value=self.params), \
             patch.object(app, "_BackgroundWorker", _ThreadWorker), \
             patch.object(app, "_stripe_runs", return_value=[]):
            ui = app.App(root)
            self.addCleanup(self.cleanup_ui, ui)
            self.addCleanup(lambda: root.after_cancel(ui._job) if ui._job else None)
            root.after_cancel(ui._job)
            ui._job = None
            with patch.object(app.filedialog, "askopenfilenames", return_value=self.paths):
                ui.choose_backgrounds()
            root.after_cancel(ui._job)
            ui._job = None
            ui.update_preview()
            self.wait_until(root, lambda: ui._preview_bg is not None and ui._future is None)
            self.assertEqual(ui._preview_bg.getpixel((8, 8)), (255, 0, 0))
            ui.vars["animate_preview"].set(True)
            ui._anim_tick()
            root.after_cancel(ui._anim_job)
            ui._anim_job = None
            self.wait_until(root, lambda: ui._preview_frame == 1)
            ui._anim_tick()
            root.after_cancel(ui._anim_job)
            ui._anim_job = None
            self.wait_until(root, lambda: ui._preview_frame == 2)
            self.assertEqual(ui._preview_bg.getpixel((8, 8)), (0, 0, 255))
            still = self.directory / "still.png"
            with patch.object(app.filedialog, "asksaveasfilename", return_value=str(still)), \
                 patch.object(app.messagebox, "showerror") as error:
                ui.save()
            error.assert_not_called()
            with Image.open(still) as image:
                self.assertEqual(image.getpixel((8, 8)), (0, 0, 255))
            ui.vars["animate_preview"].set(False)

            output = self.directory / "animation.gif"
            with patch.object(app.filedialog, "asksaveasfilename", return_value=str(output)), \
                 patch.object(app.messagebox, "showerror") as error, \
                 patch.object(app.messagebox, "showinfo") as info:
                ui.save_animation()
                self.wait_until(root, lambda: not ui._rendering)
            error.assert_not_called()
            info.assert_not_called()
            self.assertTrue(output.exists())
            for number, color in ((1, (255, 0, 0)), (2, (255, 0, 0)),
                                  (3, (0, 0, 255)), (5, (255, 0, 0))):
                with Image.open(self.directory / f"animation_{number:04d}.png") as frame:
                    self.assertEqual(frame.getpixel((8, 8)), color)

            ui.reset_background()
            self.assertEqual(ui.p.background_images, ())

    def test_invalid_selection_keeps_previous_images(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        with patch.object(app, "load_defaults", return_value=self.params), \
             patch.object(app, "_BackgroundWorker", _ThreadWorker):
            ui = app.App(root)
            self.addCleanup(self.cleanup_ui, ui)
            self.addCleanup(lambda: root.after_cancel(ui._job) if ui._job else None)
            with patch.object(app.filedialog, "askopenfilenames",
                              return_value=(str(self.directory / "missing.png"),)), \
                 patch.object(app.messagebox, "showerror") as error:
                ui.choose_backgrounds()
            error.assert_called_once()
            self.assertEqual(ui.p.background_images, self.paths)

    def test_static_background_alone_is_not_animated(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        p = replace(self.params, stripe_wobble=1, background_animated=False,
                    stripes_animated=False)
        with patch.object(app, "load_defaults", return_value=p), \
             patch.object(app, "_BackgroundWorker", _ThreadWorker):
            ui = app.App(root)
            self.addCleanup(self.cleanup_ui, ui)
            with patch.object(app.messagebox, "showinfo") as info, \
                 patch.object(app.filedialog, "asksaveasfilename") as dialog:
                ui.save_animation()
            info.assert_called_once()
            dialog.assert_not_called()
            for key in ("background_animated", "stripes_animated"):
                with self.subTest(key=key), \
                     patch.object(app.messagebox, "showinfo") as info, \
                     patch.object(app.filedialog, "asksaveasfilename", return_value="") as dialog:
                    ui.vars[key].set(True)
                    ui.save_animation()
                    info.assert_not_called()
                    dialog.assert_called_once()
                    ui.vars[key].set(False)


if __name__ == "__main__":
    unittest.main()
