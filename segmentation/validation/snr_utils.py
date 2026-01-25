"""
Signal-to-Noise Ratio (SNR) utilities for nuclear segmentation analysis.

SNR is defined as: SNR = (I_in - I_out) / std_in
where:
- I_in: mean intensity inside the nucleus
- I_out: mean intensity of the background (all pixels outside any nucleus)
- std_in: standard deviation of intensity inside the nucleus

The background intensity (I_out) is computed once per image using all pixels
where no nucleus is present (mask == 0).

Reference: https://imagej.net/plugins/trackmate/analyzers/#contrast--signalnoise-ratio

IMPORTANT: For FUCCI data with 3 channels (cyan, magenta, tubulin), only the
nuclear channels (0: cyan, 1: magenta) should be used for SNR computation.
The tubulin channel (2) is cytoplasmic and has no nuclear signal, so including
it would distort the SNR analysis. By default, this module only uses nuclear
channels (0 and 1) when computing SNR for multichannel images.

Channel convention:
- Channel 0: Cyan (G1 phase marker) - NUCLEAR
- Channel 1: Magenta (S/G2/M phase marker) - NUCLEAR
- Channel 2: Tubulin - CYTOPLASMIC (excluded from SNR by default)
"""

# Default nuclear channels for FUCCI data (cyan=0, magenta=1)
# Tubulin (channel 2) is cytoplasmic and should NOT be included in nuclear SNR
NUCLEAR_CHANNELS = (0, 1)

import numpy as np


def compute_background_intensity(image, mask, channels=None):
    """
    Compute the mean background intensity for an image.

    The background is defined as all pixels where mask == 0 (no nucleus).

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image (HxW or HxWxC).
    mask : np.ndarray
        2D label mask where 0 indicates background.
    channels : tuple of int or None
        For multichannel images, which channels to compute background for.
        If None, defaults to NUCLEAR_CHANNELS (0, 1) for 3+ channel images,
        or all channels for 2-channel images. This excludes the tubulin
        channel (2) which is cytoplasmic and has no nuclear signal.

    Returns
    -------
    float or list of float
        Mean background intensity. If image is multichannel, returns a list
        with one value per channel (only for specified channels).
    """
    background_mask = mask == 0

    if background_mask.sum() == 0:
        return None

    is_multichannel = image.ndim == 3

    if is_multichannel:
        n_channels = image.shape[-1]
        # Default: use only nuclear channels for 3+ channel images
        if channels is None:
            if n_channels >= 3:
                channels = NUCLEAR_CHANNELS
            else:
                channels = tuple(range(n_channels))
        return [image[..., c][background_mask].mean() for c in channels]
    else:
        return image[background_mask].mean()


def compute_snr_for_label(image, mask, label_id, background_intensity, channels=None):
    """
    Compute SNR for a single labeled nucleus.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image (HxW or HxWxC). If 3D, SNR is computed per channel.
    mask : np.ndarray
        2D label mask where label_id identifies the nucleus.
    label_id : int
        The label value for the nucleus of interest.
    background_intensity : float or list of float
        Precomputed background intensity for the image (must match channels).
    channels : tuple of int or None
        For multichannel images, which channels to compute SNR for.
        If None, defaults to NUCLEAR_CHANNELS (0, 1) for 3+ channel images,
        or all channels for 2-channel images. This excludes the tubulin
        channel (2) which is cytoplasmic and has no nuclear signal.

    Returns
    -------
    dict
        Dictionary with 'snr', 'mean_in', 'mean_out', 'std_in' for each channel.
        If image is 2D, returns single values. If 3D, returns lists per channel.
    """
    nucleus_mask = mask == label_id

    if nucleus_mask.sum() == 0:
        return None

    is_multichannel = image.ndim == 3

    if is_multichannel:
        n_channels = image.shape[-1]
        # Default: use only nuclear channels for 3+ channel images
        if channels is None:
            if n_channels >= 3:
                channels = NUCLEAR_CHANNELS
            else:
                channels = tuple(range(n_channels))

        results = {
            "snr": [],
            "mean_in": [],
            "mean_out": [],
            "std_in": [],
        }

        for idx, c in enumerate(channels):
            channel_img = image[..., c]
            mean_in = channel_img[nucleus_mask].mean()
            std_in = channel_img[nucleus_mask].std()
            mean_out = background_intensity[idx]  # Use index into background list

            if std_in > 0:
                snr = (mean_in - mean_out) / std_in
            else:
                snr = 0.0

            results["snr"].append(snr)
            results["mean_in"].append(mean_in)
            results["mean_out"].append(mean_out)
            results["std_in"].append(std_in)

        return results
    else:
        mean_in = image[nucleus_mask].mean()
        std_in = image[nucleus_mask].std()
        mean_out = background_intensity

        if std_in > 0:
            snr = (mean_in - mean_out) / std_in
        else:
            snr = 0.0

        return {
            "snr": snr,
            "mean_in": mean_in,
            "mean_out": mean_out,
            "std_in": std_in,
        }


def compute_snr_for_image(image, mask, channels=None):
    """
    Compute SNR for all labeled nuclei in an image.

    The background intensity is computed once for the entire image using
    all pixels where mask == 0.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image (HxW or HxWxC).
    mask : np.ndarray
        2D label mask.
    channels : tuple of int or None
        For multichannel images, which channels to use for SNR computation.
        If None, defaults to NUCLEAR_CHANNELS (0, 1) for 3+ channel images.
        This excludes the tubulin channel (2) which is cytoplasmic.

    Returns
    -------
    dict
        Dictionary mapping label_id to SNR results.
    """
    # Compute background intensity once for the entire image
    background_intensity = compute_background_intensity(image, mask, channels=channels)

    if background_intensity is None:
        return {}

    label_ids = np.unique(mask)
    label_ids = label_ids[label_ids > 0]  # Exclude background

    results = {}
    for label_id in label_ids:
        snr_result = compute_snr_for_label(
            image, mask, label_id, background_intensity, channels=channels
        )
        if snr_result is not None:
            results[int(label_id)] = snr_result

    return results


def compute_snr_for_dataset(images, masks, channel_index=None, channels=None):
    """
    Compute SNR statistics for an entire dataset.

    Parameters
    ----------
    images : list of np.ndarray
        List of images.
    masks : list of np.ndarray
        List of corresponding label masks.
    channel_index : int or None
        If not None, extract SNR for specific channel only (index into channels).
        If None and images are multichannel, returns SNR per channel.
    channels : tuple of int or None
        For multichannel images, which channels to use for SNR computation.
        If None, defaults to NUCLEAR_CHANNELS (0, 1) for 3+ channel images.
        This excludes the tubulin channel (2) which is cytoplasmic.

    Returns
    -------
    dict
        Dictionary with:
        - 'all_snr': flat list of all SNR values
        - 'per_image': list of dicts mapping label_id to SNR for each image
        - 'channel_snr': if multichannel, dict mapping channel index to SNR list
        - 'channels_used': tuple of channel indices that were used
    """
    all_snr = []
    per_image = []
    channel_snr = {}
    channels_used = None

    for img, mask in zip(images, masks):
        # Determine channels for this image
        if channels is None and img.ndim == 3:
            n_ch = img.shape[-1]
            if n_ch >= 3:
                channels_used = NUCLEAR_CHANNELS
            else:
                channels_used = tuple(range(n_ch))
        elif channels is not None:
            channels_used = channels

        img_results = compute_snr_for_image(img, mask, channels=channels)
        per_image.append(img_results)

        for label_id, result in img_results.items():
            snr_vals = result["snr"]

            if isinstance(snr_vals, list):
                # Multichannel
                if channel_index is not None:
                    all_snr.append(snr_vals[channel_index])
                else:
                    # Aggregate: use max SNR across channels as overall metric
                    all_snr.append(max(snr_vals))

                    for c, snr in enumerate(snr_vals):
                        if c not in channel_snr:
                            channel_snr[c] = []
                        channel_snr[c].append(snr)
            else:
                # Single channel
                all_snr.append(snr_vals)

    return {
        "all_snr": all_snr,
        "per_image": per_image,
        "channel_snr": channel_snr if channel_snr else None,
        "channels_used": channels_used,
    }
