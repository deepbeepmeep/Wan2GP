# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .utils import calculate_new_dimensions


class VaceImageProcessor(object):
    def __init__(self, downsample=None, seq_len=None):
        self.downsample = downsample
        self.seq_len = seq_len

    def _pillow_convert(self, image, cvt_type='RGB'):
        if image.mode != cvt_type:
            if image.mode == 'P':
                image = image.convert(f'{cvt_type}A')
            if image.mode == f'{cvt_type}A':
                bg = Image.new(cvt_type,
                               size=(image.width, image.height),
                               color=(255, 255, 255))
                bg.paste(image, (0, 0), mask=image)
                image = bg
            else:
                image = image.convert(cvt_type)
        return image

    def _load_image(self, img_path):
        if img_path is None or img_path == '':
            return None
        img = Image.open(img_path)
        img = self._pillow_convert(img)
        return img

    def _resize_crop(self, img, oh, ow, normalize=True):
        """
        Resize, center crop, convert to tensor, and normalize.
        """
        # resize and crop
        iw, ih = img.size
        if iw != ow or ih != oh:
            # resize
            scale = max(ow / iw, oh / ih)
            img = img.resize(
                (round(scale * iw), round(scale * ih)),
                resample=Image.Resampling.LANCZOS
            )
            assert img.width >= ow and img.height >= oh

            # center crop
            x1 = (img.width - ow) // 2
            y1 = (img.height - oh) // 2
            img = img.crop((x1, y1, x1 + ow, y1 + oh))

        # normalize
        if normalize:
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).unsqueeze(1)
        return img
    
    def _image_preprocess(self, img, oh, ow, normalize=True, **kwargs):
        return self._resize_crop(img, oh, ow, normalize)

    def load_image(self, data_key, **kwargs):
        return self.load_image_batch(data_key, **kwargs)

    def load_image_pair(self, data_key, data_key2, **kwargs):
        return self.load_image_batch(data_key, data_key2, **kwargs)

    def load_image_batch(self, *data_key_batch, normalize=True, seq_len=None, **kwargs):
        seq_len = self.seq_len if seq_len is None else seq_len
        imgs = []
        for data_key in data_key_batch:
            img = self._load_image(data_key)
            imgs.append(img)
        w, h = imgs[0].size
        dh, dw = self.downsample[1:]

        # compute output size
        scale = min(1., np.sqrt(seq_len / ((h / dh) * (w / dw))))
        oh = int(h * scale) // dh * dh
        ow = int(w * scale) // dw * dw
        assert (oh // dh) * (ow // dw) <= seq_len
        imgs = [self._image_preprocess(img, oh, ow, normalize) for img in imgs]
        return *imgs, (oh, ow)


class VaceVideoProcessor(object):
    def __init__(self, downsample, min_area, max_area, min_fps, max_fps, zero_start, seq_len, keep_last, **kwargs):
        self.downsample = downsample
        self.min_area = min_area
        self.max_area = max_area
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.zero_start = zero_start
        self.keep_last = keep_last
        self.seq_len = seq_len
        assert seq_len >= min_area / (self.downsample[1] * self.downsample[2])

    @staticmethod
    def resize_crop(video: torch.Tensor, oh: int, ow: int):
        """
        Resize, center crop and normalize for decord loaded video (torch.Tensor type)

        Parameters:
          video - video to process (torch.Tensor): Tensor from `reader.get_batch(frame_ids)`, in shape of (T, H, W, C)
          oh - target height (int)
          ow - target width (int)

        Returns:
            The processed video (torch.Tensor): Normalized tensor range [-1, 1], in shape of (C, T, H, W)

        Raises:
        """
        # permute ([t, h, w, c] -> [t, c, h, w])
        video = video.permute(0, 3, 1, 2)

        # resize and crop
        ih, iw = video.shape[2:]
        if ih != oh or iw != ow:
            # resize
            scale = max(ow / iw, oh / ih)
            video = F.interpolate(
                video,
                size=(round(scale * ih), round(scale * iw)),
                mode='bicubic',
                antialias=True
            )
            assert video.size(3) >= ow and video.size(2) >= oh

            # center crop
            x1 = (video.size(3) - ow) // 2
            y1 = (video.size(2) - oh) // 2
            video = video[:, :, y1:y1 + oh, x1:x1 + ow]

        # permute ([t, c, h, w] -> [c, t, h, w]) and normalize
        video = video.transpose(0, 1).float().div_(127.5).sub_(1.)
        return video

    def _video_preprocess(self, video, oh, ow):
        return self.resize_crop(video, oh, ow)

    def _get_frameid_bbox_default(self, fps, frame_timestamps, h, w, crop_box, rng):
        target_fps = min(fps, self.max_fps)
        duration = frame_timestamps[-1].mean()
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        df, dh, dw = self.downsample

        # min/max area of the [latent video]
        min_area_z = self.min_area / (dh * dw)
        max_area_z = min(self.seq_len, self.max_area / (dh * dw), (h // dh) * (w // dw))

        # sample a frame number of the [latent video]
        rand_area_z = np.square(np.power(2, rng.uniform(
            np.log2(np.sqrt(min_area_z)),
            np.log2(np.sqrt(max_area_z))
        )))
        of = min(
            (int(duration * target_fps) - 1) // df + 1,
            int(self.seq_len / rand_area_z)
        )

        # deduce target shape of the [latent video]
        target_area_z = min(max_area_z, int(self.seq_len / of))
        oh = round(np.sqrt(target_area_z * ratio))
        ow = int(target_area_z / oh)
        of = (of - 1) * df + 1
        oh *= dh
        ow *= dw

        # sample frame ids
        target_duration = of / target_fps
        begin = 0. if self.zero_start else rng.uniform(0, duration - target_duration)
        timestamps = np.linspace(begin, begin + target_duration, of)
        frame_ids = np.argmax(np.logical_and(
            timestamps[:, None] >= frame_timestamps[None, :, 0],
            timestamps[:, None] < frame_timestamps[None, :, 1]
        ), axis=1).tolist()
        return frame_ids, (x1, x2, y1, y2), (oh, ow), target_fps


        
    def _get_frameid_bbox_adjust_last(self, fps, video_frames_count, canvas_height, canvas_width, h, w,  fit_into_canvas, crop_box, rng, max_frames= 0, start_frame =0):
        from wan.utils.utils import resample

        target_fps = self.max_fps

        frame_ids= resample(fps, video_frames_count, max_frames, target_fps, start_frame )

        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        oh, ow = calculate_new_dimensions(canvas_height, canvas_width, h, w, fit_into_canvas)

        return frame_ids, (x1, x2, y1, y2), (oh, ow), target_fps

    def _get_frameid_bbox(self, fps, video_frames_count, h, w, crop_box, rng, max_frames= 0, start_frame= 0, canvas_height = 0, canvas_width = 0, fit_into_canvas= True):
        if self.keep_last:
            return self._get_frameid_bbox_adjust_last(fps, video_frames_count, canvas_height, canvas_width, h, w, fit_into_canvas, crop_box, rng, max_frames= max_frames, start_frame= start_frame)
        else:
            return self._get_frameid_bbox_default(fps, video_frames_count, h, w, crop_box, rng, max_frames= max_frames)

    def load_video(self, data_key, crop_box=None, seed=2024, **kwargs):
        return self.load_video_batch(data_key, crop_box=crop_box, seed=seed, **kwargs)

    def load_video_pair(self, data_key, data_key2, crop_box=None, seed=2024, **kwargs):
        return self.load_video_batch(data_key, data_key2, crop_box=crop_box, seed=seed, **kwargs)

    def load_video_batch(self, *data_key_batch, crop_box=None, seed=2024, max_frames= 0, trim_video =0, start_frame = 0, canvas_height = 0, canvas_width = 0, fit_into_canvas = False, **kwargs):
        rng = np.random.default_rng(seed + hash(data_key_batch[0]) % 10000)
        # read video
        import decord
        decord.bridge.set_bridge('torch')
        readers = []
        src_video = None
        for data_k in data_key_batch:
            if torch.is_tensor(data_k):
                src_video = data_k
            else:
                reader = decord.VideoReader(data_k)
                readers.append(reader)

        if src_video != None:
            fps = 16
            length = src_video.shape[0] + start_frame
            if len(readers) > 0:
                min_readers = min([len(r) for r in readers])
                length = min(length, min_readers )
        else:
            fps = readers[0].get_avg_fps()
            length = min([len(r) for r in readers])
        # frame_timestamps = [readers[0].get_frame_timestamp(i) for i in range(length)]
        # frame_timestamps = np.array(frame_timestamps, dtype=np.float32)
        max_frames = min(max_frames, trim_video) if trim_video > 0 else max_frames
        if src_video != None:
            src_video = src_video[:max_frames]
            h, w = src_video.shape[1:3]
        else:
            h, w = readers[0].next().shape[:2]
        frame_ids, (x1, x2, y1, y2), (oh, ow), fps = self._get_frameid_bbox(fps, length, h, w, crop_box, rng,  canvas_height = canvas_height, canvas_width = canvas_width, fit_into_canvas = fit_into_canvas,  max_frames=max_frames, start_frame = start_frame )

        # preprocess video
        videos = [reader.get_batch(frame_ids)[:, y1:y2, x1:x2, :] for reader in readers]
        if src_video != None:
            videos = [src_video] + videos
        videos = [self._video_preprocess(video, oh, ow) for video in videos]
        return *videos, frame_ids, (oh, ow), fps
        # return videos if len(videos) > 1 else videos[0]


def prepare_source(src_video, src_mask, src_ref_images, num_frames, image_size, device):
    for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
        if sub_src_video is None and sub_src_mask is None:
            src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
            src_mask[i] = torch.ones((1, num_frames, image_size[0], image_size[1]), device=device)
    for i, ref_images in enumerate(src_ref_images):
        if ref_images is not None:
            for j, ref_img in enumerate(ref_images):
                if ref_img is not None and ref_img.shape[-2:] != image_size:
                    canvas_height, canvas_width = image_size
                    ref_height, ref_width = ref_img.shape[-2:]
                    white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                    scale = min(canvas_height / ref_height, canvas_width / ref_width)
                    new_height = int(ref_height * scale)
                    new_width = int(ref_width * scale)
                    resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                    top = (canvas_height - new_height) // 2
                    left = (canvas_width - new_width) // 2
                    white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                    src_ref_images[i][j] = white_canvas
    return src_video, src_mask, src_ref_images
