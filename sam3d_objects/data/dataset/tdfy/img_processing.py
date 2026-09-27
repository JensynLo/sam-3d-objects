# Copyright (c) Meta Platforms, Inc. and affiliates.


import torch.nn.functional as F



# pad image to be centered for unprojecting depth
def pad_to_square_centered(image, value=0, pointmap=None):
    h, w = image.shape[-2], image.shape[-1]  # Assuming image is in (B, C, H, W) format
    if h == w:
        if pointmap is not None:
            return image, pointmap
        return image  # The image is already square

    # Calculate the padding
    diff = abs(h - w)
    pad1 = diff // 2
    pad2 = diff - pad1

    # Pad the image to make it square
    if h > w:
        padding = (pad1, pad2, 0, 0)  # Pad width (left, right, top, bottom)
    else:
        padding = (0, 0, pad1, pad2)  # Pad height
    # Apply padding to image
    padded_image = F.pad(image, padding, mode="constant", value=value)

    # Apply padding to pointmap if provided
    if pointmap is not None:
        # Pad pointmap using torch functional with NaN fill value
        padded_pointmap = F.pad(pointmap, padding, mode="constant", value=float("nan"))

        return padded_image, padded_pointmap
    return padded_image


