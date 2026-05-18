# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Output processor for Depth Anything 3.

This module handles model output processing, including tensor-to-numpy conversion,
batch dimension removal, and Prediction object creation.
"""

from __future__ import annotations

import numpy as np
import torch
from addict import Dict as AddictDict

from ...specs import Prediction


class OutputProcessor:
    """
    Output processor for converting model outputs to Prediction objects.

    Handles tensor-to-numpy conversion, batch dimension removal,
    and creates structured Prediction objects with proper data types.
    """

    def __init__(self) -> None:
        """Initialize the output processor."""

    @staticmethod
    def _numpy(tensor: torch.Tensor) -> np.ndarray:
        if tensor.is_floating_point() and tensor.dtype not in (torch.float32, torch.float64):
            tensor = tensor.float()
        return tensor.cpu().numpy()

    def __call__(self, model_output: dict[str, torch.Tensor]) -> Prediction:
        """
        Convert model output to Prediction object.

        Args:
            model_output: Model output dictionary containing depth, conf, extrinsics, intrinsics
                         Expected shapes: depth (B, N, 1, H, W), conf (B, N, 1, H, W),
                         extrinsics (B, N, 4, 4), intrinsics (B, N, 3, 3)

        Returns:
            Prediction: Object containing depth estimation results with shapes:
                       depth (N, H, W), conf (N, H, W), extrinsics (N, 4, 4), intrinsics (N, 3, 3)
        """
        # Extract data from batch dimension (B=1, N=number of images)
        depth = self._extract_depth(model_output)
        conf = self._extract_conf(model_output)
        extrinsics = self._extract_extrinsics(model_output)
        intrinsics = self._extract_intrinsics(model_output)
        sky = self._extract_sky(model_output)
        aux = self._extract_aux(model_output)
        gaussians = model_output.get("gaussians", None)
        scale_factor = model_output.get("scale_factor", None)

        return Prediction(
            depth=depth,
            sky=sky,
            conf=conf,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            is_metric=getattr(model_output, "is_metric", 0),
            gaussians=gaussians,
            aux=aux,
            scale_factor=scale_factor,
        )

    def _extract_depth(self, model_output: dict[str, torch.Tensor]) -> np.ndarray:
        """
        Extract depth tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Depth array with shape (N, H, W)
        """
        depth = self._numpy(model_output["depth"].squeeze(0).squeeze(-1))  # (N, H, W)
        return depth

    def _extract_conf(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        """
        Extract confidence tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Confidence array with shape (N, H, W) or None
        """
        conf = model_output.get("depth_conf", None)
        if conf is not None:
            conf = self._numpy(conf.squeeze(0))  # (N, H, W)
        return conf

    def _extract_extrinsics(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        """
        Extract extrinsics tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Extrinsics array with shape (N, 4, 4) or None
        """
        extrinsics = model_output.get("extrinsics", None)
        if extrinsics is not None:
            extrinsics = self._numpy(extrinsics.squeeze(0))  # (N, 4, 4)
        return extrinsics

    def _extract_intrinsics(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        """
        Extract intrinsics tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Intrinsics array with shape (N, 3, 3) or None
        """
        intrinsics = model_output.get("intrinsics", None)
        if intrinsics is not None:
            intrinsics = self._numpy(intrinsics.squeeze(0))  # (N, 3, 3)
        return intrinsics

    def _extract_sky(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        """
        Extract sky tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Sky mask array with shape (N, H, W) or None
        """
        sky = model_output.get("sky", None)
        if sky is not None:
            sky = self._numpy(sky.squeeze(0)) >= 0.5  # (N, H, W)
        return sky

    def _extract_aux(self, model_output: dict[str, torch.Tensor]) -> AddictDict:
        """
        Extract auxiliary data from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Dictionary containing auxiliary data
        """
        aux = model_output.get("aux", None)
        ret = AddictDict()
        if aux is not None:
            for k in aux.keys():
                if isinstance(aux[k], torch.Tensor):
                    ret[k] = self._numpy(aux[k].squeeze(0))
                else:
                    ret[k] = aux[k]
        return ret


# Backward compatibility alias
OutputAdapter = OutputProcessor
