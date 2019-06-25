import numpy as np
import matplotlib.pyplot as plt


from .coloration import convert_color


def get_uvw_indices(uvw_string):
    uvw_dict = {
        'u': slice(0, 1),
        'v': slice(1, 2),
        'w': slice(2, 3),
        'uv': slice(0, 2),
        'uw': slice(0, 3, 2),
        'vw': slice(1, 3),
        'uvw': slice(0, 3)
    }
    return uvw_dict.get(uvw_string)


def get_plot_slice_data(tensor, axis=2, idx=20, plot_uvw='uv', color_scheme='hv'):
    """Returns the data for the plot_slice call"""
    data = tensor.data.numpy()
    data = np.take(np.moveaxis(
        data[get_uvw_indices(plot_uvw)], 0, -1), idx, axis=axis)
    data = convert_color(data, color_scheme)
    return data


def plot_slice(tensor, filename, axis=2, idx=20, plot_uvw='uv', color_scheme='hv'):
    """Plots a slice of a 3D tensor along an axis at an index

    :param tensor: 3D tensor of uvw data
    :param filename: full filename to save the plot
    :param axis: axis to slice along. Defaults to 2, the z-axis
    :param idx: index to use for the slice. Defaults to 20
    :param plot_uvw: which wind directions to plot. Should be a string 
        possibly containing 'u', 'v', and 'w' in order
    :param color_scheme: a valid color scheme, as given in coloration
    """
    data = get_plot_slice_data(
        tensor, axis=axis, idx=idx, plot_uvw=plot_uvw, color_scheme=color_scheme
    )

    plt.imshow(data)
    plt.axis('off')
    print("Saving plot to %s" % filename)
    plt.savefig(filename)
    plt.close()


def plot_slice_comparison(tensors, filename, axis=2, idx=20, plot_uvw='uv', color_scheme='hv'):
    """Plots a side-by-side comparison of tensors

    :param tensors: a list of 3D tensors of uvw data
    :param filename: full filename to save the plot
    :param axis: axis to slice along. Defaults to 2, the z-axis
    :param idx: index to use for the slice. Defaults to 20
    :param plot_uvw: which wind directions to plot. Should be a string 
        possibly containing 'u', 'v', and 'w' in order
    :param color_scheme: a valid color scheme, as given in coloration
    """

    fig, axes = plt.subplots(nrows=1, ncols=len(tensors))

    for ax, tensor in zip(axes, tensors):
        data = get_plot_slice_data(
            tensor, axis=axis, idx=idx, plot_uvw=plot_uvw, color_scheme=color_scheme
        )
        ax.imshow(data)
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0, hspace=0)
    print("Saving plot to %s" % filename)
    plt.savefig(filename)
    plt.close()
