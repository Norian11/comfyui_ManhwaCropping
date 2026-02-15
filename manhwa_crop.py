import os
import json
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


def _snap_size_to_16(size: int) -> int:
    snapped = int(round(float(size) / 16.0) * 16.0)
    return max(16, snapped)


def _extract_square_with_white_padding(image_tensor: torch.Tensor, x: int, y: int, side: int) -> torch.Tensor:
    b, h, w, c = image_tensor.shape
    out = torch.ones((b, side, side, c), dtype=image_tensor.dtype, device=image_tensor.device)

    src_x0 = max(0, int(x))
    src_y0 = max(0, int(y))
    src_x1 = min(int(w), int(x) + int(side))
    src_y1 = min(int(h), int(y) + int(side))

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - int(x)
    dst_y0 = src_y0 - int(y)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    out[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = image_tensor[:, src_y0:src_y1, src_x0:src_x1, :]
    return out


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
                "x": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
                "y": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
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

        side = _snap_size_to_16(int(size))
        x0 = int(x)
        y0 = int(y)
        cropped = _extract_square_with_white_padding(image_tensor, x0, y0, side)
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
                "x": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
                "y": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
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
            side = _snap_size_to_16(int(size))
            side = min(side, int(patch.shape[0]), int(patch.shape[1]))

            x0 = int(x)
            y0 = int(y)
            x1 = x0 + side
            y1 = y0 + side

            dst_x0 = max(0, x0)
            dst_y0 = max(0, y0)
            dst_x1 = min(w, x1)
            dst_y1 = min(h, y1)

            if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                src_x0 = dst_x0 - x0
                src_y0 = dst_y0 - y0
                src_x1 = src_x0 + (dst_x1 - dst_x0)
                src_y1 = src_y0 + (dst_y1 - dst_y0)
                base[dst_y0:dst_y1, dst_x0:dst_x1, :] = patch[src_y0:src_y1, src_x0:src_x1, :]
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
