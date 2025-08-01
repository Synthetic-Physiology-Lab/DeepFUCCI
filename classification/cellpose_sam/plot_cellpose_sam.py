import fastremap
import matplotlib.pyplot as plt
import numpy as np
from cellpose import utils
from natsort import natsorted


cl_colors = [
    [0, 255, 255],  # cyan
    [150, 75, 0],  # brown
    [255, 0, 255],
]  # magenta


def fig5(save_fig=False):
    aps = []
    errors = []
    dat = np.load("deepfucci_cellposeSAM.npy", allow_pickle=True).item()
    aps.append(np.array(dat["aps"]))
    errors.append(np.array(dat["errors"]))

    aps = np.array(aps)
    errors = np.array(errors)
    print(np.nanmean(errors, axis=1), np.nanmean(errors, axis=(1, 2)))
    print(np.nanmean(aps, axis=1), np.nanmean(aps, axis=(1, 2)))
    img_files = dat["img_files"]

    folders = natsorted(np.unique([f.name.split("_")[0] for f in img_files]))

    colors_tab = plt.get_cmap("tab10").colors
    colors = [[0.8, 0.8, 0.3], [0.5, 0.5, 0.5], [0.7, 0.5, 1]]

    from scipy.stats import mode

    class_colors_pred = np.minimum(1, cl_colors.copy() / 255 + 0.2)
    class_colors_pred[-1] = np.minimum(1, class_colors_pred[-1] + 0.1)
    class_colors_true = np.maximum(0, cl_colors.copy() / 255 - 0.2)
    classes = dat["classes"]
    classes_true = dat["classes_true"]
    masks_pred = dat["masks_pred"]
    imgs = dat["imgs"]
    masks_true = dat["masks_true"]
    fig = plt.figure(figsize=(14 * 2.0 / 3, 5), dpi=150)
    grid = plt.GridSpec(
        3, 7, hspace=0.0, wspace=0.1, top=0.95, bottom=0.05, left=0.01, right=0.99
    )
    il = 0
    iexs = [0, 1, 2]
    for i, iex in enumerate(iexs):
        iap = [i for i, folder in enumerate(folders) if folder in img_files[iex].name][
            0
        ]
        ax = plt.subplot(grid[i // 3, i % 3])
        if i == 0:
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1] - 0.1 * pos[3], pos[2], 0.4 * pos[3]])
        class0 = classes[iex].copy()
        class_true = classes_true[iex].copy()
        masks_pred0 = masks_pred[iex].copy()
        masks_true0 = masks_true[iex].copy()
        masks_pred0[class0 == 0] = 0

        masks_pred0 = fastremap.renumber(masks_pred0)[0]
        masks_true0 = fastremap.renumber(masks_true0)[0]

        cid = (
            np.array(
                [
                    mode(class0[masks_pred0 == j])[0]
                    for j in range(1, masks_pred0.max() + 1)
                ]
            )
            - 1
        )
        tid = (
            np.array(
                [
                    mode(class_true[masks_true0 == j])[0]
                    for j in range(1, masks_true0.max() + 1)
                ]
            )
            - 1
        )
        cid = cid.astype("int")
        tid = tid.astype("int")

        ax.imshow(imgs[iex])
        outlines = utils.outlines_list(masks_true0)
        for j, outline in enumerate(outlines):
            ax.plot(
                outline[:, 0],
                outline[:, 1],
                color=class_colors_true[tid[j]],
                lw=1.5,
                ls="-",
            )

        if 1:
            outlines = utils.outlines_list(masks_pred0)
            for j, outline in enumerate(outlines):
                ax.plot(
                    outline[:, 0],
                    outline[:, 1],
                    color=class_colors_pred[cid[j]],
                    lw=2.5,
                    ls="--",
                    dashes=(1.5, 2),
                )

        ax.axis("off")
        plt.show()

    for k in range(2):
        eps = errors if k == 0 else aps
        vp = plt.violinplot(
            np.nanmean(eps[0], axis=-1),
            showmeans=True,
            showmedians=False,
            showextrema=False,
            positions=[j],
        )
        vp["bodies"][0].set_facecolor(colors[j])
        vp["bodies"][0].set_alpha(0.35)
        plt.plot(
            j + 0.3 * np.array([-1, 1]),
            np.nanmean(eps[j], axis=-1).mean() * np.ones(2),
            color=colors[j],
            lw=3,
        )
        if k == 0:
            plt.ylabel("error rate @ 0.5 IoU")
            plt.ylim([0.0, 0.8])
            plt.yticks(np.arange(0, 0.85, 0.2))
        else:
            plt.ylabel("AP @ 0.5 IoU")
            plt.ylim([0.0, 1])
            plt.yticks(np.arange(0, 1.05, 0.2))
        plt.show()
