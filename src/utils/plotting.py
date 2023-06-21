from pathlib import Path
from typing import Any, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output, display
from matplotlib import animation, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from src.utils.utils import is_notebook

# Set maximum buffer size for animation
mpl.rcParams["animation.embed_limit"] = 2**128


class Artist:
    margin = 0.2
    figure = None
    imgs = list()

    def save_img(self, filename: Union[Path, str], dirname: Union[Path, str] = "."):
        if dirname != "." or isinstance(dirname, str):
            dirname = Path(dirname)
        dirname.mkdir(parents=True, exist_ok=True)
        self.figure.save(fp=(dirname / filename).with_suffix(".png"), format="PNG")

    def save_gif_from_files(
        self, filename: Union[Path, str], dirname: Union[Path, str] = ".", fps=20
    ):
        files = Path(dirname).glob("**/*.png")
        self.imgs = [Image.open(file) for file in sorted(files)]
        self.save_gif(filename, dirname, fps)

    def save_gif(self, filename: Union[Path, str], dirname: Union[Path, str] = ".", fps=20):
        if dirname != "." or isinstance(dirname, str):
            dirname = Path(dirname)
        dirname.mkdir(parents=True, exist_ok=True)
        self.imgs[0].save(
            fp=(dirname / filename).with_suffix(".gif"),
            format="GIF",
            append_images=self.imgs[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
        )
        self.imgs.clear()

    def append_img(self):
        self.imgs.append(self.figure)

    def show(self):
        if is_notebook():
            clear_output(wait=True)
            display(self.figure)
        else:
            self.figure.show()

    def plot_ode(
        self,
        ts,
        solution,
        prediction=None,
        title="Trajectory of Damped Harmonic Oscillator",
        *args: Any,
        **kwds: Any,
    ):
        fig, ax = plt.subplots(
            figsize=kwds.get("figsize", None)
        )  # Create a figure and axis objects
        ax.plot(ts, solution, "b", label="q(t)")  # Plot the solution
        if prediction is not None:  # if prediction is given, plot it
            ax.plot(ts, prediction, "r", linestyle="dashed", label="prediction")
        xp = kwds.get("xp", None)
        if xp is not None:
            ax.scatter(xp[0], xp[1], marker="x", label=xp[2])
        ax.axis(
            (
                ts.min() - self.margin,
                ts.max() + self.margin,
                solution.min() - self.margin,
                solution.max() + self.margin,
            )
        )
        ax.grid(axis="both", which="both", linestyle="dashed")  # Add grid lines to the plot
        ax.update({"xlabel": "t", "ylabel": "q", "title": title})
        ax.legend()  # Add legend to the plot
        fig.tight_layout()
        self.figure = self.figure_to_image()

    def plot_pde(
        self,
        ts,
        xs,
        solution,
        prediction=None,
        title: str = "1D viscous Burgers' equation",
        *args: Any,
        **kwds: Any,
    ):
        fig, ax = plt.subplots(
            figsize=kwds.get("figsize", None)
        )  # Create a figure and axis objects

        # Additional margin for title
        extra_top_margin: float = 0.08

        # Additional space for preds / error plots
        ncol: int = 1
        width_margin: float = 0
        if prediction is not None:
            ncol = 3
            width_margin = 0.5

        gs0 = gridspec.GridSpec(1, ncol)
        gs0.update(
            top=1 - 0.06 - extra_top_margin,
            bottom=1 - 1.0 / 2.0 + 0.06,
            left=0.15,
            right=0.85,
            wspace=width_margin,
        )

        # t = 0, 25, 50, 75(%)
        snapshot_idx = torch.arange(0, len(ts), step=len(ts) * 0.25, dtype=int)

        # Row 0: Plot Solution (+ Prediction & Error)
        imshow_config = dict(
            interpolation="nearest",
            cmap="rainbow",
            extent=[ts.min(), ts.max(), xs.min(), xs.max()],
            origin="lower",
            aspect="auto",
        )

        # Draw vertical line at time to visualize
        ax = plt.subplot(gs0[0, 0])
        line = torch.linspace(xs.min(), xs.max(), 2)[:, None]
        for t in ts[snapshot_idx]:
            ax.plot(t * torch.ones((2, 1)), line, "w-", linewidth=1)

        # Plot collocation points
        xp = kwds.get("xp", None)
        if xp is not None:
            ax.plot(
                xp[:, 0],
                xp[:, 1],
                "kx",
                label="Data (%d points)" % (len(xp)),
                markersize=4,  # marker size doubled
                clip_on=False,
                alpha=0.5,
            )
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.9, -0.05),
                ncol=5,
                frameon=False,
                prop={"size": 15},
            )

        imgs = [solution]
        ax_titles = ["Solution"]
        if prediction is not None:
            imgs += [prediction, abs(solution - prediction)]
            ax_titles += ["Prediction", "Absolute Error"]

        for col, img, ax_title in zip(range(ncol), imgs, ax_titles):
            ax = plt.subplot(gs0[0, col])
            if col == 0:
                ax.set_ylabel("$x$")
            ax.update({"xlabel": "$t$", "title": ax_title})
            h = ax.imshow(img, **imshow_config)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=12)

        # Row 1: Plot solution at certain timestamps(t=0.00, 0.25, 0.50, 0.75)(%)
        ncol = len(snapshot_idx)
        gs1 = gridspec.GridSpec(1, ncol)
        gs1.update(top=1 - 1.0 / 2.0 - 0.1, bottom=0.10, wspace=0.4)

        for col, idx in zip(range(ncol), snapshot_idx):
            ax = plt.subplot(gs1[0, col])
            ax.plot(xs, solution[:, idx], "b", label="Solution")
            if prediction is not None:
                ax.plot(
                    xs,
                    prediction[:, idx],
                    "r",
                    linestyle="dashed",
                    label="Prediction",
                )
            ax.set_xlabel("$x$")
            if col == 0:
                ax.set_ylabel("$u(t,x)$")
            ax.set_title("$t=%.2fs$" % (ts[idx]), fontsize=8)
            ax.axis("square")
            ax.axis(
                (
                    xs.min() - self.margin,
                    xs.max() + self.margin,
                    solution.min() - self.margin,
                    solution.max() + self.margin,
                )
            )
            ax.tick_params(axis="both", which="both", labelsize=6)
            ax.grid(which="both", axis="both", linestyle="--")

        if prediction is not None:
            ax = plt.subplot(gs1[0, 0])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

        fig.suptitle(title)
        self.figure = self.figure_to_image()

    def figure_to_image(self) -> Image:
        """Converts the current figure in matplotlib to an RGB image.

        Returns:
            img (Image): The converted image.
        """
        # Get the current figure's canvas
        canvas = plt.get_current_fig_manager().canvas

        # Draw the canvas
        canvas.draw()

        # Convert the canvas to an RGB image
        img = Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())

        # Clear the current figure and all plots
        plt.cla()
        plt.clf()
        plt.close("all")

        return img
