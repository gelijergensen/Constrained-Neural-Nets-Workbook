import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Used only for the projection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .coloration import convert_color
from .two_dimensional import get_plot_slice_data


def plot_quiver_of_3D_tensor(tensor, filename, color_scheme="sph"):
    """Plots a 3D quiver plot of the tensor in color"""

    data = tensor.data.numpy()

    len_x, len_y, len_z = data.shape[1:]
    xyz = np.mgrid[0:len_x, 0:len_y, 0:len_z]

    color = np.moveaxis(data, 0, -1)
    color = (color - np.min(color)) / (np.max(color) - np.min(color))

    color = convert_color(color, color_scheme)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.quiver(
        xyz[0],
        xyz[1],
        xyz[2],
        data[0],
        data[1],
        data[2],
        length=0.3,
        normalize=True,
        colors=color.reshape(-1, 3),
    )
    ax.set_axis_off()
    print("Saving plot to %s" % filename)
    plt.savefig(filename)
    plt.close()


def animation_of_slice_comparison(
    time_tensors,
    filename,
    axis=2,
    idx=20,
    plot_uvw="uv",
    color_scheme="hv",
    dpi=100,
    fps=20,
):
    """Plots a series of tensors over time

    :param time_tensors: a list of list of tensors to plot, where the first list
        is the time dimension and the second is the comparison tensors
    :param filename: full filename to save the plot
    :param axis: axis to slice along. Defaults to 2, the z-axis
    :param idx: index to use for the slice. Defaults to 20
    :param plot_uvw: which wind directions to plot. Should be a string
        possibly containing 'u', 'v', and 'w' in order
    :param color_scheme: a valid color scheme, as given in coloration
    """

    fig, axes = plt.subplots(nrows=1, ncols=len(time_tensors[0]))
    ln = list()
    for ax, tensor in zip(axes, time_tensors[0]):
        data = get_plot_slice_data(
            tensor,
            axis=axis,
            idx=idx,
            plot_uvw=plot_uvw,
            color_scheme=color_scheme,
        )
        ln.append(ax.imshow(data))
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0, hspace=0)

    def update(frame_idx):
        for i, (ax, tensor) in enumerate(zip(axes, time_tensors[frame_idx])):
            # if i == 1:
            #     print(tensor.max())
            data = get_plot_slice_data(
                tensor,
                axis=axis,
                idx=idx,
                plot_uvw=plot_uvw,
                color_scheme=color_scheme,
            )
            ln[i].set_data(data)
        return ln

    print("Saving animation to %s" % filename)
    ani = FuncAnimation(
        fig, update, frames=range(0, len(time_tensors)), blit=True
    )
    fig.set_dpi(dpi)
    ani.save(filename, writer="pillow", fps=fps)
    plt.close()


def animation_of_3D_tensor(tensor, filename, idx=20):

    data = tensor.select(dim=-1, index=idx).data.numpy()
    data = np.moveaxis(data, 0, -1)
    # data = rescale_data(data)

    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    recolored = convert_color(data, "hsv")

    fig = plt.figure()
    ax = fig.gca()
    ax.set_axis_off()
    ln = ax.imshow(data)

    def init():
        return (ln,)

    def update(alpha):

        alpha = (np.cos(alpha) + 1) / 2

        to_plot = data * alpha + recolored * (1 - alpha)
        ln.set_data(to_plot)
        return (ln,)

    print("Saving animation to %s" % filename)
    ani = FuncAnimation(
        fig,
        update,
        frames=np.linspace(0, 2 * np.pi, num=200),
        init_func=init,
        blit=True,
    )
    fig.set_dpi(100)
    ani.save(filename, writer="pillow", fps=20)
    plt.close()
