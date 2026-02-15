import os
import json
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


def _snap_size_to_16(size: int, max_side: int) -> int:
    if max_side <= 0:
        return 1
    max_multiple = (max_side // 16) * 16
    if max_multiple >= 16:
        snapped = int(round(size / 16.0) * 16)
        return max(16, min(snapped, max_multiple))
    return max(1, min(int(size), max_side))


class ManhwaCrop:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        files = sorted(files)
        if not files:
            files = [""]
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "x": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "size": ("INT", {"default": 512, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {
                "image_in": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("crop", "original", "x", "y", "size")
    FUNCTION = "crop"
    CATEGORY = "image/transform"

    def _load_image_from_input(self, image_name: str) -> torch.Tensor:
        image_path = folder_paths.get_annotated_filepath(image_name)
        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        arr = np.array(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None,]

    def crop(self, image: str, x: int, y: int, size: int, image_in: torch.Tensor = None):
        if image_in is None:
            image_tensor = self._load_image_from_input(image)
        else:
            image_tensor = image_in

        if image_tensor.ndim != 4:
            raise ValueError("Expected IMAGE tensor with shape [B,H,W,C].")

        _, h, w, _ = image_tensor.shape
        side = _snap_size_to_16(int(size), min(int(w), int(h)))
        max_x = max(0, int(w) - side)
        max_y = max(0, int(h) - side)
        x0 = max(0, min(int(x), max_x))
        y0 = max(0, min(int(y), max_y))

        cropped = image_tensor[:, y0:y0 + side, x0:x0 + side, :]
        return (cropped, image_tensor, x0, y0, side)


class ManhwaStitchSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.compress_level = 4
        self._last_trigger = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "modified_square": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "size": ("INT", {"default": 512, "min": 1, "max": 100000, "step": 1}),
                "filename_prefix": ("STRING", {"default": "ManhwaCrop"}),
                "stitch_and_save_trigger": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "saved_file")
    FUNCTION = "stitch_and_save"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def stitch_and_save(
        self,
        original_image,
        modified_square,
        x,
        y,
        size,
        filename_prefix="ManhwaCrop",
        stitch_and_save_trigger=0,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        key = str(unique_id) if unique_id is not None else "_default"
        trigger = int(stitch_and_save_trigger)
        last = self._last_trigger.get(key, None)
        should_save = (last is not None) and (trigger != last)
        self._last_trigger[key] = trigger

        b = original_image.shape[0]
        stitched_batch = []
        saved_paths = []

        for idx in range(b):
            base = original_image[idx].clone()
            patch = modified_square[idx if idx < modified_square.shape[0] else 0]

            h = int(base.shape[0])
            w = int(base.shape[1])
            side_limit = min(w, h, int(patch.shape[0]), int(patch.shape[1]))
            side = _snap_size_to_16(int(size), side_limit)
            x0 = max(0, min(int(x), w - side))
            y0 = max(0, min(int(y), h - side))
            base[y0:y0 + side, x0:x0 + side, :] = patch[:side, :side, :]
            stitched_batch.append(base.unsqueeze(0))

        stitched = torch.cat(stitched_batch, dim=0)
        ui_images = []

        if should_save:
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, stitched[0].shape[1], stitched[0].shape[0]
            )
            for batch_number, image in enumerate(stitched):
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for k in extra_pnginfo:
                            metadata.add_text(k, json.dumps(extra_pnginfo[k]))

                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                output_path = os.path.join(full_output_folder, file)
                img.save(output_path, pnginfo=metadata, compress_level=self.compress_level)
                saved_paths.append(output_path)
                ui_images.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                })
                counter += 1

        # Always show the incoming modified square preview in this node UI.
        preview_name = f"manhwa_square_{int(time.time() * 1000)}.png"
        preview_path = os.path.join(self.temp_dir, preview_name)
        preview_img = modified_square[0 if modified_square.shape[0] > 0 else 0]
        p = 255.0 * preview_img.cpu().numpy()
        Image.fromarray(np.clip(p, 0, 255).astype(np.uint8)).save(preview_path, compress_level=1)
        ui_images.insert(0, {
            "filename": preview_name,
            "subfolder": "",
            "type": "temp",
        })

        saved_file = saved_paths[0] if saved_paths else ""
        return {"ui": {"images": ui_images}, "result": (stitched, saved_file)}


NODE_CLASS_MAPPINGS = {
    "Manhwa Crop": ManhwaCrop,
    "Manhwa Stitch Save": ManhwaStitchSave,
}
