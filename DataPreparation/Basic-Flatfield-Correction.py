from basicpy import BaSiC
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
import os
import numpy as np
import yaml
import shutil

plot_correction = False


def apply_basic_correction(image, channel):
    print("Loading frames")
    frames = image.get_image_data("TYX", C=int(channels[channel]))
    basic = BaSiC(get_darkfield=False)
    print("Fitting frames")
    basic.fit(frames)
    if os.path.isdir(f"basic_model_{channel}"):
        shutil.rmtree(f"basic_model_{channel}")
    basic.save_model(f"basic_model_{channel}")

    print("Plotting correction")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    im = axes[0].imshow(basic.flatfield, cmap="gray")
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Flatfield")
    im = axes[1].imshow(basic.darkfield, cmap="gray")
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("Darkfield")
    axes[2].plot(basic.baseline)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Baseline")
    fig.tight_layout()
    plt.savefig(f"basic_figures/Basic_Result_{channel}.png")
    plt.close()

    if plot_correction:
        # False to not have baseline correction!
        images_transformed = basic.transform(frames, timelapse=False)
        vmin_frames = np.min(frames)
        vmax_frames = np.max(frames)
        vmin_trafo = np.min(images_transformed)
        vmax_trafo = np.max(images_transformed)
        for i in range(frames.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            im = axes[0].imshow(frames[i], vmin=vmin_frames, vmax=vmax_frames, cmap="gray")
            fig.colorbar(im, ax=axes[0])
            axes[0].set_title("Original")
            im = axes[1].imshow(images_transformed[i],
                                vmin=vmin_trafo,
                                vmax=vmax_trafo,
                                cmap="gray")
            fig.colorbar(im, ax=axes[1])
            axes[1].set_title("Corrected")
            fig.tight_layout()
            plt.savefig(f"basic_figures/basic_frame_{channel}_{i}.png")
            plt.close()


if not os.path.exists("basic_figures"):
    os.mkdir("basic_figures")

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)
filename = metadata["filename"]
image = AICSImage(filename)

channels = metadata["channels"]
print("Channels: ", channels)


for channel in ["cyan", "magenta", "tubulin", "actin"]:
    print("Basic correction for: ", channel)
    apply_basic_correction(image, channel)
