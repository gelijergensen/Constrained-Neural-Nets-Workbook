import numpy as np
from matplotlib.colors import hsv_to_rgb


def rescale_spherical(data):
    data[:, :, 0] = np.tanh(data[:, :, 0])
    data[:, :, 1] = (data[:, :, 1] + np.pi) / np.pi / 2
    data[:, :, 2] = data[:, :, 2] / np.pi
    return data


def signed_magnitude_rescale(data):
    """Rescales (-inf, inf) to (0, 1)"""
    return (np.tanh(data) + 1.0) / 2.0


def magnitude_rescale(data):
    """Rescales [0, inf) to [0, 1)"""
    return np.tanh(np.log(data + 1))
    # return data / data.max()


def angle_rescale(data, min, max):
    """Rescales [min, max] to [0, 1]"""
    return (data - min) / (max - min)


def cartesian_to_polar(data):
    """Converts [..., (x,y)] to [..., (r,theta)]. Assumes that the last
    dimension of data has length 2"""

    out = np.zeros(data.shape)

    out[..., 0] = np.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2)
    out[..., 1] = np.arctan2(data[..., 1], data[..., 0])
    return out


def cartesian_to_spherical(data):
    out = np.zeros(data.shape)

    out[..., 0] = np.sqrt(
        data[..., 0] ** 2 + data[..., 1] ** 2 + data[..., 2] ** 2
    )
    # for elevation angle defined from Z-axis down
    out[..., 1] = np.arctan2(data[..., 1], data[..., 0])
    # This (azimuth) will become hue
    out[..., 2] = np.arctan2(
        np.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2), data[..., 2]
    )
    return out


def spherical_to_hsv(data):
    return data[::-1]


def polar_to_hsv(data):
    """Converts [..., (r,theta)] to [..., ((theta + pi)/(2pi), tanh(r), tanh(r)]
    """
    out = np.zeros(data.shape[:-1] + (3,))
    out[..., 0] = angle_rescale(data[..., 1], min=-np.pi, max=np.pi)
    out[..., 1] = magnitude_rescale(data[..., 0])
    out[..., 2] = out[..., 1]
    return out


def grayscale_to_rbg(data):
    return np.tile(data, 3)


def convert_color(data, color="rgb"):
    """Converts rgb data to a different color scheme

    : param data: a numpy array to modify
    : param color: a color scheme. Should be one of "rbg", "hsv", "sph"
    """

    # 3D
    if color == "hsv":
        return hsv_to_rgb(data)
    elif color == "sph":
        return hsv_to_rgb(
            spherical_to_hsv(rescale_spherical(cartesian_to_spherical(data)))
        )
    # 2D
    elif color == "hv":
        return hsv_to_rgb(polar_to_hsv(cartesian_to_polar(data)))
    # 1D
    elif color == "gray":
        return grayscale_to_rbg(data)
    else:
        return data
