# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F

from .dinov2 import vision_transformer as vits

# (arch, num_register_tokens) of the torch.hub entrypoints used by the released configs.
# kwargs mirror facebookresearch/dinov2 hub/backbones.py.
_DINO_MODELS = {
    "dinov2_vitl14_reg": ("vit_large", 4),
    "dinov2_vitl14": ("vit_large", 0),
    "dinov2_vitb14_reg": ("vit_base", 4),
    "dinov2_vitb14": ("vit_base", 0),
}


def build_dinov2(dino_model: str) -> torch.nn.Module:
    """Build the DINOv2 architecture from the vendored source (no torch.hub, no
    network). Weights are NOT loaded here: the SAM 3D generator checkpoints in
    ./ckpts already contain the full backbone weights and overwrite everything."""
    arch, num_register_tokens = _DINO_MODELS[dino_model]
    return vits.__dict__[arch](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=num_register_tokens > 0,
        interpolate_offset=0.0 if num_register_tokens > 0 else 0.1,
    )


class Dino(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        dino_model: str = "dinov2_vitb14",
        normalize_images: bool = True,
        prenorm_features: bool = False,
    ):
        super().__init__()
        self.backbone = build_dinov2(dino_model)
        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.register_buffer("mean", torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1), persistent=False)
        self.register_buffer("std", torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1), persistent=False)
        self.requires_grad_(False)
        self.eval()

    def _preprocess_input(self, x):
        _resized_images = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode="bilinear",
            align_corners=False,
        )

        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)

        if self.normalize_images:
            _resized_images = _resized_images.sub_(self.mean).div_(self.std)

        return _resized_images

    def _forward_last_layer(self, input_img):
        output = self.backbone.forward_features(input_img)
        if self.prenorm_features:
            features = output["x_prenorm"]
            tokens = F.layer_norm(features, features.shape[-1:])
        else:
            tokens = torch.cat(
                [
                    output["x_norm_clstoken"].unsqueeze(1),
                    output["x_norm_patchtokens"],
                ],
                dim=1,
            )
        return tokens

    def forward(self, x, **kwargs):
        _resized_images = self._preprocess_input(x)
        tokens = self._forward_last_layer(_resized_images)
        return tokens.to(x.dtype)
