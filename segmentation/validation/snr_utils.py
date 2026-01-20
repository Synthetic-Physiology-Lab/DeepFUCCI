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
"""

import numpy as np


def compute_background_intensity(image, mask):
    """
    Compute the mean background intensity for an image.

    The background is defined as all pixels where mask == 0 (no nucleus).

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image (HxW or HxWxC).
    mask : np.ndarray
        2D label mask where 0 indicates background.

    Returns
    -------
    float or list of float
        Mean background intensity. If image is multichannel, returns a list
        with one value per channel.
    """
    background_mask = mask == 0

    if background_mask.sum() == 0:
        return None

    is_multichannel = image.ndim == 3

    if is_multichannel:
        n_channels = image.shape[-1]
        return [image[..., c][background_mask].mean() for c in range(n_channels)]
    else:
        return image[background_mask].mean()


def compute_snr_for_label(image, mask, label_id, background_intensity):
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
        Precomputed background intensity for the image.

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
        results = {
            "snr": [],
            "mean_in": [],
            "mean_out": [],
            "std_in": [],
        }

        for c in range(n_channels):
            channel_img = image[..., c]
            mean_in = channel_img[nucleus_mask].mean()
            std_in = channel_img[nucleus_mask].std()
            mean_out = background_intensity[c]

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


def compute_snr_for_image(image, mask):
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

    Returns
    -------
    dict
        Dictionary mapping label_id to SNR results.
    """
    # Compute background intensity once for the entire image
    background_intensity = compute_background_intensity(image, mask)

    if background_intensity is None:
        return {}

    label_ids = np.unique(mask)
    label_ids = label_ids[label_ids > 0]  # Exclude background

    results = {}
    for label_id in label_ids:
        snr_result = compute_snr_for_label(image, mask, label_id, background_intensity)
        if snr_result is not None:
            results[int(label_id)] = snr_result

    return results


def compute_snr_for_dataset(images, masks, channel_index=None):
    """
    Compute SNR statistics for an entire dataset.

    Parameters
    ----------
    images : list of np.ndarray
        List of images.
    masks : list of np.ndarray
        List of corresponding label masks.
    channel_index : int or None
        If not None, extract SNR for specific channel only.
        If None and images are multichannel, returns SNR per channel.

    Returns
    -------
    dict
        Dictionary with:
        - 'all_snr': flat list of all SNR values
        - 'per_image': list of dicts mapping label_id to SNR for each image
        - 'channel_snr': if multichannel, dict mapping channel index to SNR list
    """
    all_snr = []
    per_image = []
    channel_snr = {}

    for img, mask in zip(images, masks):
        img_results = compute_snr_for_image(img, mask)
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
    }
