# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os
import re
from threading import Thread

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from ..logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
RANK = int(os.getenv("RANK", "0"))

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def load_resource_as_video_frames(
    resource_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    video_loader_type="ffmpeg",
):
    """
    Load video frames from either a video or an image (as a single-frame video).
    Alternatively, if input is a list of PIL images, convert its format
    """
    if isinstance(resource_path, list):
        img_mean = torch.tensor(img_mean, dtype=torch.float16, device="cpu")[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float16, device="cpu")[:, None, None]
        assert all(isinstance(img_pil, Image.Image) for img_pil in resource_path)
        assert len(resource_path) is not None
        orig_height, orig_width = resource_path[0].size
        orig_height, orig_width = (
            orig_width,
            orig_height,
        )  # For some reason, this method returns these swapped
        images = []
        for img_pil in resource_path:
            img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
            assert img_np.dtype == np.uint8, "np.uint8 is expected for JPEG images"
            img_np = img_np / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1)
            # float16 precision should be sufficient for image tensor storage
            img = img.to(dtype=torch.float16)
            # normalize by mean and std
            img -= img_mean
            img /= img_std
            images.append(img)
        images = torch.stack(images)
        if not offload_video_to_cpu:
            images = images.cuda()
        return images, orig_height, orig_width

    is_image = (
        isinstance(resource_path, str)
        and os.path.splitext(resource_path)[-1].lower() in IMAGE_EXTS
    )
    if is_image:
        return load_image_as_single_frame_video(
            image_path=resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
        )
    else:
        return load_video_frames(
            video_path=resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )


def load_image_as_single_frame_video(
    image_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
):
    """Load an image as a single-frame video."""
    images, image_height, image_width = _load_img_as_tensor(image_path, image_size)
    images = images.unsqueeze(0).half()

    img_mean = torch.tensor(img_mean, dtype=torch.float16, device="cpu")[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16, device="cpu")[:, None, None]
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, image_height, image_width


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    video_loader_type="ffmpeg",
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    assert isinstance(video_path, str)
    if video_path.startswith("<load-dummy-video"):
        # Check for pattern <load-dummy-video-N> where N is an integer
        match = re.match(r"<load-dummy-video-(\d+)>", video_path)
        num_frames = int(match.group(1)) if match else 60
        return load_dummy_video(image_size, offload_video_to_cpu, num_frames=num_frames)
    elif video_path.startswith("<load-zero-video"):
        # Check for pattern <load-zero-video-N> where N is an integer
        match = re.match(r"<load-zero-video-(\d+)>", video_path)
        num_frames = int(match.group(1)) if match else 60
        return load_dummy_video(
            image_size, offload_video_to_cpu, num_frames=num_frames, do_zeros=True
        )
    elif os.path.isdir(video_path):
        return load_video_frames_from_image_folder(
            image_folder=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
        )
    elif os.path.splitext(video_path)[-1].lower() in VIDEO_EXTS:
        return load_video_frames_from_video_file(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
    else:
        # No recognized extension (e.g., extensionless OIL paths) — attempt video loading.
        # Only raise if the loader itself fails to decode frames.
        try:
            return load_video_frames_from_video_file(
                video_path=video_path,
                image_size=image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                img_mean=img_mean,
                img_std=img_std,
                async_loading_frames=async_loading_frames,
                video_loader_type=video_loader_type,
            )
        except Exception as e:
            raise NotImplementedError(
                f"Only video files and image folders are supported; "
                f"failed to load '{video_path}' as video: {e}"
            ) from e


def load_video_frames_from_image_folder(
    image_folder,
    image_size,
    offload_video_to_cpu,
    img_mean,
    img_std,
    async_loading_frames,
):
    """
    Load the video frames from a directory of image files ("<frame_index>.<img_ext>" format)
    """
    frame_names = [
        p
        for p in os.listdir(image_folder)
        if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
    ]
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.<img_ext>"
        logger.warning(
            f'frame names are not in "<frame_index>.<img_ext>" format: {frame_names[:5]=}, '
            f"falling back to lexicographic sort."
        )
        frame_names.sort()
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {image_folder}")
    img_paths = [os.path.join(image_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncImageFrameLoader(
            img_paths, image_size, offload_video_to_cpu, img_mean, img_std
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    # float16 precision should be sufficient for image tensor storage
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)
    video_height, video_width = None, None
    for n, img_path in enumerate(
        tqdm(img_paths, desc=f"frame loading (image folder) [rank={RANK}]")
    ):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_video_frames_from_video_file(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean,
    img_std,
    async_loading_frames,
    gpu_acceleration=False,
    gpu_device=None,
    video_loader_type="ffmpeg",
):
    """Load the video frames from a video file."""
    if video_loader_type == "ffmpeg":
        return load_video_frames_from_video_file_using_ffmpeg(
            video_path=video_path,
            image_size=image_size,
            img_mean=img_mean,
            img_std=img_std,
            offload_video_to_cpu=offload_video_to_cpu,
        )
    if video_loader_type == "cv2":
        return load_video_frames_from_video_file_using_cv2(
            video_path=video_path,
            image_size=image_size,
            img_mean=img_mean,
            img_std=img_std,
            offload_video_to_cpu=offload_video_to_cpu,
        )
    raise RuntimeError("video_loader_type must be either 'ffmpeg' or 'cv2'")


def load_video_frames_from_video_file_using_ffmpeg(
    video_path: str,
    image_size: int,
    img_mean: tuple = (0.5, 0.5, 0.5),
    img_std: tuple = (0.5, 0.5, 0.5),
    offload_video_to_cpu: bool = False,
) -> torch.Tensor:
    from shared.utils.video_decode import decode_video_frames_ffmpeg, probe_video_stream_metadata

    metadata = probe_video_stream_metadata(video_path)
    if metadata is None:
        raise RuntimeError(f"Unable to probe video metadata for {video_path}")
    num_frames = int(metadata.get("frame_count") or 0)
    if num_frames <= 0:
        raise RuntimeError(f"Unable to determine frame count for {video_path}")

    frames = decode_video_frames_ffmpeg(video_path, 0, num_frames, target_fps=None, bridge="torch")
    if frames.numel() == 0:
        raise RuntimeError(f"No frames could be decoded from video: {video_path}")

    video_tensor = frames.permute(0, 3, 1, 2).float()
    if video_tensor.shape[-2:] != (image_size, image_size):
        video_tensor = F.interpolate(video_tensor, size=(image_size, image_size), mode="bicubic", align_corners=False)
    video_tensor = video_tensor.half()
    video_tensor /= 255

    img_mean = torch.tensor(img_mean, dtype=torch.float16).view(1, 3, 1, 1)
    img_std = torch.tensor(img_std, dtype=torch.float16).view(1, 3, 1, 1)
    if not offload_video_to_cpu:
        video_tensor = video_tensor.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    video_tensor -= img_mean
    video_tensor /= img_std
    return video_tensor, metadata["display_height"], metadata["display_width"]


def load_video_frames_from_video_file_using_cv2(
    video_path: str,
    image_size: int,
    img_mean: tuple = (0.5, 0.5, 0.5),
    img_std: tuple = (0.5, 0.5, 0.5),
    offload_video_to_cpu: bool = False,
) -> torch.Tensor:
    """
    Load video from path, convert to normalized tensor with specified preprocessing

    Args:
        video_path: Path to video file
        image_size: Target size for square frames (height and width)
        img_mean: Normalization mean (RGB)
        img_std: Normalization standard deviation (RGB)

    Returns:
        torch.Tensor: Preprocessed video tensor in shape (T, C, H, W) with float16 dtype
    """
    import cv2  # delay OpenCV import to avoid unnecessary dependency

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = num_frames if num_frames > 0 else None

    frames = []
    pbar = tqdm(desc=f"frame loading (OpenCV) [rank={RANK}]", total=num_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC
        )
        frames.append(frame_resized)
        pbar.update(1)
    cap.release()
    pbar.close()

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames could be decoded from video: {video_path}. "
            f"The file may be corrupted, empty, or encoded with an unsupported codec."
        )

    # Convert to tensor
    frames_np = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W, C)
    video_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)

    img_mean = torch.tensor(img_mean, dtype=torch.float16).view(1, 3, 1, 1)
    img_std = torch.tensor(img_std, dtype=torch.float16).view(1, 3, 1, 1)
    if not offload_video_to_cpu:
        video_tensor = video_tensor.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    video_tensor -= img_mean
    video_tensor /= img_std
    return video_tensor, original_height, original_width


def load_dummy_video(image_size, offload_video_to_cpu, num_frames=60, do_zeros=False):
    """
    Load a dummy video with random frames for testing and compilation warmup purposes.
    """
    video_height, video_width = 480, 640  # dummy original video sizes
    if not do_zeros:
        images = torch.randn(num_frames, 3, image_size, image_size, dtype=torch.float16)
    else:
        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)
    if not offload_video_to_cpu:
        images = images.cuda()
    return images, video_height, video_width


def _load_img_as_tensor(img_path, image_size):
    """Load and resize an image and convert it into a PyTorch tensor."""
    img = Image.open(img_path).convert("RGB")
    orig_width, orig_height = img.width, img.height
    img = TF.resize(img, size=(image_size, image_size))
    img = TF.to_tensor(img)
    return img, orig_height, orig_width


class AsyncImageFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self._images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(
                    range(len(self.images)),
                    desc=f"frame loading (image folder) [rank={RANK}]",
                ):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # float16 precision should be sufficient for image tensor storage
        img = img.to(dtype=torch.float16)
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.cuda()
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)
