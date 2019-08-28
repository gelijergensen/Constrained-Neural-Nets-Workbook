"""A layer for constructing a spectral representation of a function"""

import numpy as np
import torch
from torch import nn


class SpectralReconstruction(nn.Module):
    """A layer which computes the spectral coefficients of a "low" resolution
    function profile, reconstructs the function as a Fourier Series, and then
    evaluates the resulting function at a set of points"""

    def __init__(self, signal_ndim):
        super().__init__()
        self.signal_ndim = signal_ndim
        if signal_ndim > 1:
            raise NotImplementedError("Signal dimension > 1 not yet supported")

    def forward(self, fn_profile, query_points):
        """Applies the DFT to compute a Fourier Series of the gridded fn_profile
        (a set of "y-values" corresponding to one period of some function) and 
        uses the Fourier Series to evaluate the query points. NOTE: the query
        points are evaluated as if the function has a period on precisely the
        interval [0, 1]. This means that the query points probably need to be
        normalized to that interval when used in a neural network

        WARNING: Presently only d=1 is supported! TODO: upgrade

        :param fn_profile: A tensor of shape (B, N_1, ..., N_d), where d is 
            the number of signal dimensions and B is the batch size
        :param query_points: A tensor of shape (B, N_q, d), a batch, B, of N_q
            d-dimensional query points
        :returns: fn(query_points), where fn is the implicit function defined
            by the function profile. Will have shape (B, N_q)
        """
        coefs = torch.rfft(fn_profile, self.signal_ndim, normalized=False)
        # batched outer product
        angles = (
            2
            * np.pi
            * torch.einsum(
                "i,bqd->bqid",
                torch.arange(coefs.size()[1], dtype=fn_profile.dtype),
                query_points,
            )
        )
        print(f"fn_profile: {fn_profile.size()}")
        print(f"coefs: {coefs.size()}")
        print(f"query_points: {query_points.size()}")
        print(f"angles: {angles.size()}")
        # reconstruct as a sum of sines and cosines
        sum_dims = tuple(
            range(2, len(angles.size()))
        )  # all dims but batch and query
        print(
            f"coefs[..., 0] reshaped: {coefs[..., 0].unsqueeze(1).unsqueeze(-1).size()}"
        )
        cosines = torch.mean(
            coefs[..., 0].unsqueeze(1).unsqueeze(-1) * torch.cos(angles),
            dim=sum_dims,
        )
        sines = torch.mean(
            coefs[..., 1].unsqueeze(1).unsqueeze(-1) * torch.sin(angles),
            dim=sum_dims,
        )
        return cosines - sines
