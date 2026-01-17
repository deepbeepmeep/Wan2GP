import os
import cv2
import requests
import torch
import numpy as np
import PIL.Image as Image
import PIL.ImageOps
# from insightface.app import FaceAnalysis
# from facexlib.parsing import init_parsing_model
from torchvision.transforms.functional import normalize
from typing import Union, Optional
from models.hyvideo.data_kits.face_align import AlignImage
from shared.utils import files_locator as fl 


def _img2tensor(img: np.ndarray, bgr2rgb: bool = True) -> torch.Tensor:
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def _pad_to_square(img: np.ndarray, pad_color: int = 255) -> np.ndarray:
    h, w, _ = img.shape
    if h == w:
        return img

    if h > w:
        pad_size = (h - w) // 2
        padded_img = cv2.copyMakeBorder(
            img,
            0,
            0,
            pad_size,
            h - w - pad_size,
            cv2.BORDER_CONSTANT,
            value=[pad_color] * 3,
        )
    else:
        pad_size = (w - h) // 2
        padded_img = cv2.copyMakeBorder(
            img,
            pad_size,
            w - h - pad_size,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[pad_color] * 3,
        )

    return padded_img


class FaceProcessor:
    def __init__(self):
        self.align_instance = AlignImage("cuda", det_path= fl.locate_file("det_align/detface.pt"))
        self.align_instance.facedet.model.to("cpu")


    def process(
        self,
        image: Union[str, PIL.Image.Image],
        resize_to: int = 512,
        border_thresh: int = 10,
        face_crop_scale: float = 1.5,
        remove_bg= False,
        # area=1.25
    ) -> PIL.Image.Image:

        image_pil = PIL.ImageOps.exif_transpose(image).convert("RGB")
        w, h = image_pil.size
        self.align_instance.facedet.model.to("cuda")        
        _, _, bboxes_list = self.align_instance(np.array(image_pil)[:,:,[2,1,0]], maxface=True)
        self.align_instance.facedet.model.to("cpu")        

        try:
            bboxSrc = bboxes_list[0]
        except:
            bboxSrc = [0, 0, w, h]
        x1, y1, ww, hh = bboxSrc
        x2, y2 = x1 + ww, y1 + hh
        # ww, hh = (x2-x1) * area, (y2-y1) * area
        # center = [(x2+x1)//2, (y2+y1)//2]
        # x1 = max(center[0] - ww//2, 0)
        # y1 = max(center[1] - hh//2, 0)
        # x2 = min(center[0] + ww//2, w)
        # y2 = min(center[1] + hh//2, h)
  
        frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape
        image_to_process = None

        is_close_to_border = (
            x1 <= border_thresh
            and y1 <= border_thresh
            and x2 >= w - border_thresh
            and y2 >= h - border_thresh
        )

        if is_close_to_border:
            # print(
            #     "[Info] Face is close to border, padding original image to square."
            # )
            image_to_process = _pad_to_square(frame, pad_color=255)
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            side = int(max(x2 - x1, y2 - y1) * face_crop_scale)
            half = side // 2

            left = int(max(cx - half, 0))
            top = int(max(cy - half, 0))
            right = int(min(cx + half, w))
            bottom = int(min(cy + half, h))

            cropped_face = frame[top:bottom, left:right]
            image_to_process = _pad_to_square(cropped_face, pad_color=255)

        image_resized = cv2.resize(
            image_to_process, (resize_to, resize_to), interpolation=cv2.INTER_LANCZOS4 # .INTER_AREA
        )

        face_tensor = _img2tensor(image_resized).to("cpu")

        from shared.utils.utils import remove_background, convert_tensor_to_image
        if remove_bg:
            face_tensor = remove_background(face_tensor)
        img_out = Image.fromarray(face_tensor.clone().mul_(255).permute(1,2,0).to(torch.uint8).cpu().numpy())
        return img_out


# class FaceProcessor2:
#     def __init__(self, antelopv2_path=".", device: Optional[torch.device] = None):
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = device

#         providers = (
#             ["CUDAExecutionProvider"]
#             if self.device.type == "cuda"
#             else ["CPUExecutionProvider"]
#         )
#         self.app = FaceAnalysis(
#             name="antelopev2", root=antelopv2_path, providers=providers
#         )
#         self.app.prepare(ctx_id=0, det_size=(640, 640))

#         self.parsing_model = init_parsing_model(
#             model_name="bisenet", device=self.device
#         )
#         self.parsing_model.eval()

#         print("FaceProcessor initialized successfully.")

#     def process(
#         self,
#         image: Union[str, PIL.Image.Image],
#         resize_to: int = 512,
#         border_thresh: int = 10,
#         face_crop_scale: float = 1.5,
#         extra_input: bool = False,
#     ) -> PIL.Image.Image:
#         if isinstance(image, str):
#             if image.startswith("http://") or image.startswith("https://"):
#                 image = PIL.Image.open(requests.get(image, stream=True, timeout=10).raw)
#             elif os.path.isfile(image):
#                 image = PIL.Image.open(image)
#             else:
#                 raise ValueError(
#                     f"Input string is not a valid URL or file path: {image}"
#                 )
#         elif not isinstance(image, PIL.Image.Image):
#             raise TypeError(
#                 "Input must be a file path, a URL, or a PIL.Image.Image object."
#             )

#         image = PIL.ImageOps.exif_transpose(image).convert("RGB")

#         frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         faces = self.app.get(frame)
#         h, w, _ = frame.shape
#         image_to_process = None

#         if not faces:
#             print(
#                 "[Warning] No face detected. Using the whole image, padded to square."
#             )
#             image_to_process = _pad_to_square(frame, pad_color=255)
#         else:
#             largest_face = max(
#                 faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
#             )
#             x1, y1, x2, y2 = map(int, largest_face.bbox)

#             is_close_to_border = (
#                 x1 <= border_thresh
#                 and y1 <= border_thresh
#                 and x2 >= w - border_thresh
#                 and y2 >= h - border_thresh
#             )

#             if is_close_to_border:
#                 print(
#                     "[Info] Face is close to border, padding original image to square."
#                 )
#                 image_to_process = _pad_to_square(frame, pad_color=255)
#             else:
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#                 side = int(max(x2 - x1, y2 - y1) * face_crop_scale)
#                 half = side // 2

#                 left = max(cx - half, 0)
#                 top = max(cy - half, 0)
#                 right = min(cx + half, w)
#                 bottom = min(cy + half, h)

#                 cropped_face = frame[top:bottom, left:right]
#                 image_to_process = _pad_to_square(cropped_face, pad_color=255)

#         image_resized = cv2.resize(
#             image_to_process, (resize_to, resize_to), interpolation=cv2.INTER_AREA
#         )

#         face_tensor = (
#             _img2tensor(image_resized, bgr2rgb=True).unsqueeze(0).to(self.device)
#         )
#         with torch.no_grad():
#             normalized_face = normalize(face_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             parsing_out = self.parsing_model(normalized_face)[0]
#             parsing_mask = parsing_out.argmax(dim=1, keepdim=True)

#         background_mask_np = (parsing_mask.squeeze().cpu().numpy() == 0).astype(
#             np.uint8
#         )
#         white_background = np.ones_like(image_resized, dtype=np.uint8) * 255
#         mask_3channel = cv2.cvtColor(background_mask_np * 255, cv2.COLOR_GRAY2BGR)
#         result_img_bgr = np.where(mask_3channel == 255, white_background, image_resized)
#         result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
#         img_white_bg = PIL.Image.fromarray(result_img_rgb)
#         if extra_input:
#             # 2. Create image with transparent background (new logic)
#             # Create an alpha channel: 255 for foreground (not background), 0 for background
#             alpha_channel = (parsing_mask.squeeze().cpu().numpy() != 0).astype(
#                 np.uint8
#             ) * 255

#             # Convert the resized BGR image to RGB
#             image_resized_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

#             # Stack RGB channels with the new alpha channel
#             rgba_image = np.dstack((image_resized_rgb, alpha_channel))

#             # Create PIL image from the RGBA numpy array
#             img_transparent_bg = PIL.Image.fromarray(rgba_image, "RGBA")

#             return img_white_bg, img_transparent_bg
#         else:
#             return img_white_bg
