# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU RGB preprocessing and prompt expansion for image and sampled video inputs."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from models.glm5_3_flash.vision.config import VISION_MERGE, VISION_PATCH, VISION_TEMPORAL_PATCH


@dataclass(frozen=True)
class VisualInput:
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    marker: str
    prompt_fragment: str


def _canvas(frames, height, width, minimum, maximum):
    factor = VISION_PATCH * VISION_MERGE
    unit = VISION_TEMPORAL_PATCH * factor * factor
    aligned_frames = max(VISION_TEMPORAL_PATCH, math.ceil(frames / VISION_TEMPORAL_PATCH) * VISION_TEMPORAL_PATCH)

    def align(value):
        return math.ceil(value / factor) * factor

    h, w = align(height), align(width)
    if aligned_frames * h * w < minimum * unit:
        scale = math.sqrt(minimum * unit / (aligned_frames * height * width))
        h, w = align(math.ceil(height * scale)), align(math.ceil(width * scale))
    if aligned_frames * h * w > maximum * unit:
        if aligned_frames * factor * factor > maximum * unit:
            raise ValueError("visual token budget cannot fit one spatial merge group per temporal patch")
        low, high = 1, height
        h, w = factor, factor
        while low <= high:
            candidate = (low + high) // 2
            ch, cw = align(candidate), align(max(1, math.floor(width * candidate / height)))
            if aligned_frames * ch * cw <= maximum * unit:
                h, w, low = ch, cw, candidate + 1
            else:
                high = candidate - 1
    return h, w


def _rgb_tensor(image):
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        import numpy as np
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                array = np.array(source.convert("RGB"), copy=True)
        else:
            array = np.array(image.convert("RGB"), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    if tensor.ndim != 3 or tensor.shape[0] != 3 or tensor.dtype != torch.uint8:
        raise ValueError("visual inputs must be RGB uint8 [3, height, width] tensors or Pillow images")
    if min(tensor.shape[1:]) <= 0:
        raise ValueError("visual dimensions must be positive")
    return tensor.contiguous()


class VisionProcessor:
    """Apply the checkpoint's fixed patch geometry, resize budget and CLIP normalization."""

    def __init__(self, checkpoint):
        root = Path(checkpoint.root if hasattr(checkpoint, "root") else checkpoint)
        config = json.loads((root / "processor_config.json").read_text())
        self.config = {}
        for kind in ("image", "video"):
            value = config[kind + "_processor"]
            for name, expected in (("patch_size", VISION_PATCH), ("merge_size", VISION_MERGE),
                                   ("temporal_patch_size", VISION_TEMPORAL_PATCH), ("patch_expand_factor", 1)):
                if value.get(name) != expected:
                    raise ValueError(f"{kind}_processor.{name} must be {expected}")
            if value.get("do_rescale", True) is not True or value.get("do_normalize", True) is not True:
                raise ValueError("vision preprocessing requires rescaling and CLIP normalization")
            if value.get("do_resize", True) is not True:
                raise ValueError("vision preprocessing requires the configured resize budget")
            self.config[kind] = value

    def _pixels(self, frames, kind):
        config = self.config[kind]
        count, _, height, width = frames.shape
        minimum, maximum = int(config["min_image_tokens"]), int(config["max_image_tokens"])
        if minimum <= 0 or maximum < minimum:
            raise ValueError("invalid visual token limits")
        budget_frames = math.ceil(count / VISION_TEMPORAL_PATCH) * VISION_TEMPORAL_PATCH
        target_h, target_w = _canvas(budget_frames, height, width, minimum, maximum)
        scale = min(target_h / height, target_w / width)
        if budget_frames * height * width >= VISION_TEMPORAL_PATCH * (VISION_PATCH * VISION_MERGE) ** 2 * minimum:
            scale = min(1.0, scale)
        content_h = max(1, min(target_h, math.floor(height * scale)))
        content_w = max(1, min(target_w, math.floor(width * scale)))
        if (content_h, content_w) != (height, width):
            frames = F.interpolate(frames, size=(content_h, content_w), mode="bicubic", align_corners=False, antialias=True)
        frames = F.pad(frames, (0, target_w - content_w, 0, target_h - content_h))
        rescale_factor = float(config.get("rescale_factor", 1 / 255))
        if not math.isfinite(rescale_factor) or rescale_factor <= 0:
            raise ValueError("visual rescaling must be positive and finite")
        mean = torch.tensor(config["image_mean"], dtype=torch.float32).reshape(1, 3, 1, 1)
        std = torch.tensor(config["image_std"], dtype=torch.float32).reshape(1, 3, 1, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("visual normalization must have finite means and positive finite standard deviations")
        pixel_scale = 1.0 / rescale_factor
        pixels = (frames.float() - mean * pixel_scale) / (std * pixel_scale)
        if kind == "image":
            pixels = pixels.expand(VISION_TEMPORAL_PATCH, -1, -1, -1)
        elif count % VISION_TEMPORAL_PATCH:
            pixels = torch.cat((pixels, pixels[-1:]), dim=0)
        grid_t = pixels.shape[0] // VISION_TEMPORAL_PATCH
        grid_h, grid_w = target_h // VISION_PATCH, target_w // VISION_PATCH
        grouped = pixels.reshape(grid_t, VISION_TEMPORAL_PATCH, 3, grid_h // VISION_MERGE,
                                 VISION_MERGE, VISION_PATCH, grid_w // VISION_MERGE, VISION_MERGE, VISION_PATCH)
        ordered = grouped.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches = ordered.reshape(grid_t * grid_h * grid_w, -1).contiguous()
        return patches, torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64)

    def image(self, image):
        pixels, grid = self._pixels(_rgb_tensor(image).unsqueeze(0), "image")
        tokens = int(grid.prod()) // VISION_MERGE ** 2
        fragment = "<|begin_of_image|>" + "<|image|>" * tokens + "<|end_of_image|>"
        return VisualInput(pixels, grid, "<|image|>", fragment)

    def video(self, frames, timestamps):
        """Process already sampled RGB frames and their timestamps in seconds."""
        frames = torch.stack([_rgb_tensor(frame) for frame in frames])
        times = torch.as_tensor(timestamps, dtype=torch.float64)
        if times.shape != (frames.shape[0],) or not torch.isfinite(times).all():
            raise ValueError("provide one finite timestamp for every sampled video frame")
        if (times < 0).any() or (times[1:] < times[:-1]).any():
            raise ValueError("video timestamps must be nonnegative and monotone")
        pixels, grid = self._pixels(frames, "video")
        tokens = int(grid[0, 1:].prod()) // VISION_MERGE ** 2
        fragments = [f"<|begin_of_image|>{'<|image|>' * tokens}<|end_of_image|>{float(time):.1f} seconds"
                     for time in times[::VISION_TEMPORAL_PATCH]]
        fragment = "<|begin_of_video|>" + "".join(fragments) + "<|end_of_video|>"
        return VisualInput(pixels, grid, "<|video|>", fragment)

    def video_file(self, path):
        """Decode a video with FFmpeg using the reference processor's timestamp sampling."""
        from fractions import Fraction
        import shutil
        import subprocess

        import numpy as np

        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("video files require ffmpeg and ffprobe; sampled RGB frames can use video()")
        path = Path(path).resolve(strict=True)
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration", "-of", "json", str(path)]
        metadata = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        if not metadata.get("streams"):
            raise ValueError("video has no decodable visual stream")
        stream = metadata["streams"][0]
        source_fps = float(Fraction(stream["avg_frame_rate"]))
        if source_fps <= 0 or not math.isfinite(source_fps):
            raise ValueError("video needs a positive finite frame rate")
        count = stream.get("nb_frames")
        if count in (None, "N/A"):
            command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                       "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)]
            counted = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            count = counted["streams"][0]["nb_read_frames"]
        count = int(count)
        if count <= 0:
            raise ValueError("video must contain at least one frame")
        duration = stream.get("duration", metadata.get("format", {}).get("duration"))
        duration = float(duration) if duration not in (None, "N/A") else round((count - 1) / source_fps) + 1
        target_fps = float(self.config["video"].get("fps", 2))
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("video duration and sampling rate must be positive and finite")
        requested = max(1, min(int(duration * target_fps), 2048))
        if count < requested:
            indices = np.linspace(0, count - 1, requested, dtype=int).tolist()
        else:
            indices, next_time = [], 0.0
            for index in range(count):
                if index / source_fps >= next_time:
                    indices.append(index)
                    next_time += 1 / target_fps
                    if next_time >= int(duration):
                        break
            if len(indices) < requested:
                begin, end = (indices[0], indices[-1]) if indices else (0, count - 1)
                indices = np.linspace(begin, end, requested, dtype=int).tolist()
            elif len(indices) > requested:
                indices = np.linspace(0, count - 1, requested, dtype=int).tolist()
        indices = list(dict.fromkeys(indices))
        if len(indices) % VISION_TEMPORAL_PATCH:
            indices.append(indices[-1])
        unique = sorted(set(indices))
        selection = "+".join(f"eq(n\\,{index})" for index in unique)
        command = ["ffmpeg", "-nostdin", "-v", "error", "-noautorotate", "-i", str(path), "-map", "0:v:0", "-vf", "select=" + selection,
                   "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        decoded = subprocess.run(command, check=True, capture_output=True).stdout
        width, height = int(stream["width"]), int(stream["height"])
        frame_bytes = width * height * 3
        if len(decoded) != len(unique) * frame_bytes:
            raise ValueError("decoded video frame count does not match the sampled timeline")
        frames = torch.from_numpy(np.frombuffer(decoded, dtype=np.uint8).copy())
        frames = frames.reshape(len(unique), height, width, 3).permute(0, 3, 1, 2)
        lookup = dict(zip(unique, frames))
        return self.video([lookup[index] for index in indices], [index / source_fps for index in indices])
