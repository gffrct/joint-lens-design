"""
    Classes that represent batched specifications, lenses and lens structures

    The lens class is used to batch lenses regardless of their structure and number of variables
    This allows the ray-tracing operations to be batched for better use of computational resources

    Underlying tensors have 2D shape (batch x max length)
        curvatures are padded with 0's
        thicknesses are padded with 0's
        refractive indices are padded with 1's

    1D compact forms can be recovered from the *_flat() methods
    2D forms can be updated from the 1D forms
"""
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import ray_tracing as rt


def mask_replace(mask, src, dst):
    assert src.shape == mask.shape
    assert len(dst.shape) == 1
    return tf.where(mask, tf.scatter_nd(tf.where(mask), dst, mask.shape), src)


def g_from_n_v(n, v):
    assert len(n.shape) == len(v.shape) == 1
    w = tf.constant([[-7.497527849096219, -7.49752916467739], [0.07842101471405442, -0.07842100095362642]])
    mean = tf.constant([[1.6426209211349487, 48.8505973815918]])

    g = (tf.stack((n, v), axis=-1) - mean) @ w

    return g


def n_v_from_g(g):
    assert len(g.shape) == 2 and g.shape[1] == 2
    w = tf.constant([[-0.06668863644654068, 6.3758429552417315], [-0.0666886481483064, -6.375841836481304]])
    mean = tf.constant([[1.6426209211349487, 48.8505973815918]])

    return tf.unstack(g @ w + mean, axis=1)


def find_valid_curvatures(sequence):
    # Conditions for exclusion: current and previous elements are air, or last curvature of the system
    previous_element = np.concatenate((np.zeros_like(sequence.mask_G[:, 0:1]), sequence.mask_G[:, :-1]), axis=1)
    valid_curvature_mask = sequence.mask_G | previous_element & sequence.mask_except_last & sequence.mask
    return valid_curvature_mask


def get_normalized_lens_variables(lens, trainable_vars, add_bfl=False, scale_factor=1):
    """
        Initialize TF variables from a lens object
        The choice of "scale_factor" only has an effect during optimization,
            by changing the relative scale of the variables w.r.t. the gradients
        With the Adam optimizer, reducing the scale_factor has an effect similar to augmenting the learning rate
    """

    # First scale the variables to get EFL == 1
    current_efl = lens.efl
    if np.isfinite(current_efl.numpy().item()):
        lens = lens.scale(1 / current_efl)
    else:
        # If using a random starting point with bad behaviour, compute the last curvature so that EFL=1
        lens.flat_c = rt.compute_last_curvature(lens.structure, lens.flat_c_but_last, lens.flat_t, lens.flat_nd)

    # We first define the glass materials by the refractive indices 'nd' and Abbe numbers 'v'
    # We go to the normalized form 'g' which is the one that we will optimize
    # Then we go back to the first form 'nd' and 'v' for ray tracing
    g = tf.Variable(lambda: g_from_n_v(lens.flat_nd, lens.flat_v) * scale_factor, name='lens_g', dtype=tf.float32,
                    trainable=trainable_vars['g'])

    t_non_flat = lens.t
    if add_bfl:
        # Find last thickness, which corresponds to the defocus
        last_t_position = lens.structure.mask.sum(axis=1) - 1
        last_t_indices = tf.stack((tf.range(len(lens)), last_t_position), axis=1)
        last_t = tf.gather_nd(lens.t, last_t_indices)

        # Remove the BFL
        updated_last_t = last_t - lens.bfl

        # Update
        t_non_flat = tf.tensor_scatter_nd_update(t_non_flat, last_t_indices, updated_last_t)
    t = tf.Variable(lambda: t_non_flat[lens.structure.mask] * scale_factor, name='lens_t', dtype=tf.float32,
                    trainable=trainable_vars['t'])

    # Curvatures are optimized as is
    # We exclude the last curvature which is computed on the fly
    # We also exclude the curvatures of the surfaces surrounded by air (usually the aperture stop)
    valid_curvatures = find_valid_curvatures(lens.structure)

    c = tf.Variable(lambda: lens.c[valid_curvatures] * scale_factor, name='lens_c', dtype=tf.float32,
                    trainable=trainable_vars['c'])

    return c, t, g


def map_glass_to_closest(g, catalog_g):
    dist = tf.norm(g[:, None, :] - catalog_g[None, :, :], axis=-1)
    min_dist_idx = tf.argmin(dist, axis=1)
    return tf.gather(catalog_g, min_dist_idx), catalog_g


def get_lens_from_normalized(structure, c, t, g, catalog_g, add_bfl=False, scale_factor=1, qc_variables=True):
    # Undo the scaling operation
    c = c / scale_factor
    t = t / scale_factor
    g = g / scale_factor

    # If quantized continuous glass variables, map the glass variables to the closest catalog glass
    if qc_variables:
        g, _ = tf.grad_pass_through(map_glass_to_closest)(g, catalog_g)

    # Retrieve the lens
    nd, v = n_v_from_g(g)
    # Fill the curvature array
    c2d = tf.zeros_like(structure.mask, dtype=c.dtype)
    c2d = mask_replace(find_valid_curvatures(structure), c2d, c)
    flat_c = c2d[structure.mask_except_last]
    # Compute the last curvature with an algebraic solve to enforce EFL = 1
    c = rt.compute_last_curvature(structure, flat_c, t, nd)
    lens = Lens(structure, c, t, nd, v)

    if add_bfl:
        # Find last thickness, which corresponds to the defocus
        last_t_position = lens.structure.mask.sum(axis=1) - 1
        last_t_indices = tf.stack((tf.range(len(lens)), last_t_position), axis=1)
        last_t_indices_flat = last_t_position + np.arange(structure.mask.shape[0]) * structure.mask.shape[1]
        last_t = t[last_t_indices_flat.tolist()]

        # Compute the new value by adding the BFL to the defocus
        updated_t = lens.bfl + last_t

        # Update
        lens.t = tf.tensor_scatter_nd_update(lens.t, last_t_indices, updated_t)
    return lens


class Structure:

    def __init__(self, stop_idx, mask=None, mask_G=None, sequence=None):
        self.stop_idx = stop_idx
        assert len(self.stop_idx.shape) == 1

        if sequence is not None:
            assert mask is None
            assert mask_G is None
            assert isinstance(sequence, np.ndarray)

            n = sequence.shape[0]
            sequence = sequence.view('U1').reshape(n, -1)

            self.mask = np.array(sequence != '')
            self.mask_G = np.array(sequence == 'G')

        else:
            assert mask is not None
            assert mask_G is not None
            self.mask = mask
            self.mask_G = mask_G

        assert len(self.mask.shape) == 2
        assert len(self.mask_G.shape) == 2

    def __len__(self):
        return self.mask.shape[0]

    def up_to_stop(self):
        """
            Returns the lens structures up to the aperture stop of the systems (used to recover the entrance pupil)
        """
        max_len = self.stop_idx.max()
        mask = np.arange(max_len)[None, :] < self.stop_idx[:, None]
        return Structure(self.stop_idx, self.mask[:, :max_len] & mask, self.mask_G[:, :max_len] & mask)

    def clone(self):
        return Structure(self.stop_idx.copy(), self.mask.copy(), self.mask_G.copy())

    def __getitem__(self, index):
        index = slice(index, index + 1) if isinstance(index, int) else index
        max_len = self.mask[index].sum(axis=1).max()
        return Structure(self.stop_idx[index], self.mask[index, :max_len], self.mask_G[index, :max_len])

    @property
    def last_g_idx(self):
        # Find the index of the last glass element
        idx = np.broadcast_to(np.arange(self.mask.shape[1], dtype=self.stop_idx.dtype), self.mask.shape)
        return np.where(self.mask_G, idx, 0).argmax(axis=1)

    @property
    def mask_except_last(self):
        mask = self.mask.copy()
        mask[np.arange(len(self)), self.last_g_idx + 1] = 0
        return mask


@dataclass
class Specs:
    structure: Structure
    epd: tf.Tensor
    hfov: tf.Tensor
    vig_up: tf.Tensor = None
    vig_down: tf.Tensor = None
    vig_x: tf.Tensor = None

    def __post_init__(self):
        assert len(self.epd.shape) == 1, 'EPD should be 1-dimensional'
        assert len(self.hfov.shape) == 1, 'HFOV should be 1-dimensional'

        if any((self.vig_up is None, self.vig_down is None)):
            self.vig_up = tf.zeros_like(self.epd)
            self.vig_down = tf.zeros_like(self.epd)
            self.vig_x = tf.zeros_like(self.epd)

    def __len__(self):
        return len(self.structure)

    def scale(self, factor):
        return Specs(self.structure, self.epd * factor, self.hfov, self.vig_up, self.vig_down, self.vig_x)

    def up_to_stop(self):
        return Specs(self.structure.up_to_stop(), self.epd, self.hfov, self.vig_up, self.vig_down, self.vig_x)

    def __getitem__(self, index):
        index = slice(index, index + 1) if isinstance(index, int) else index
        return Specs(
            self.structure[index],
            self.epd[index],
            self.hfov[index],
            self.vig_up[index],
            self.vig_down[index],
            self.vig_x[index]
        )


@dataclass
class Lens:
    structure: Structure
    c: tf.Tensor
    t: tf.Tensor
    nd: tf.Tensor
    v: tf.Tensor

    def __post_init__(self):

        if len(self.c.shape) == 1:
            flat_c = self.c
            self.c = tf.zeros_like(self.structure.mask, dtype=self.c.dtype)
            self.flat_c = flat_c

        if len(self.t.shape) == 1:
            flat_t = self.t
            self.t = tf.zeros_like(self.structure.mask, dtype=self.t.dtype)
            self.flat_t = flat_t

        if len(self.nd.shape) == 1:
            flat_nd = self.nd
            self.nd = tf.ones_like(self.structure.mask, dtype=self.nd.dtype)
            self.flat_nd = flat_nd

        if len(self.v.shape) == 1:
            flat_v = self.v
            self.v = tf.fill(self.structure.mask.shape, np.nan)
            self.flat_v = flat_v

    def __len__(self):
        return len(self.structure)

    def scale(self, factor):
        return Lens(self.structure, self.c / factor, self.t * factor, self.nd, self.v)

    def up_to_stop(self):
        structure = self.structure.up_to_stop()
        new_len = structure.mask.shape[1]
        return Lens(
            structure,
            self.c[:, :new_len][structure.mask],
            self.t[:, :new_len][structure.mask],
            self.nd[:, :new_len][structure.mask_G],
            self.v[:, :new_len][structure.mask_G],
        )

    def __getitem__(self, index):
        index = slice(index, index + 1) if isinstance(index, int) else index
        structure = self.structure[index]
        max_length = structure.mask.shape[1]
        return Lens(
            structure,
            self.c[index, :max_length],
            self.t[index, :max_length],
            self.nd[index, :max_length],
            self.v[index, :max_length]
        )

    def detach(self):
        return Lens(self.structure, tf.stop_gradient(self.c), tf.stop_gradient(self.t),
                    tf.stop_gradient(self.nd), tf.stop_gradient(self.v))

    @property
    def flat_c(self):
        return self.c[self.structure.mask]

    @flat_c.setter
    def flat_c(self, c):
        self.c = mask_replace(self.structure.mask, self.c, c)

    @property
    def flat_c_but_last(self):
        c_mask = self.structure.mask.copy()
        c_mask[np.arange(len(self)), self.structure.mask.sum(axis=1) - 1] = False
        return self.c[c_mask]

    @property
    def flat_t(self):
        return self.t[self.structure.mask]

    @flat_t.setter
    def flat_t(self, t):
        self.t = mask_replace(self.structure.mask, self.t, t)

    @property
    def flat_nd(self):
        return self.nd[self.structure.mask_G]

    @flat_nd.setter
    def flat_nd(self, nd):
        self.nd = mask_replace(self.structure.mask_G, self.nd, nd)

    @property
    def flat_v(self):
        return self.v[self.structure.mask_G]

    @flat_v.setter
    def flat_v(self, v):
        self.v = mask_replace(self.structure.mask_G, self.v, v)

    def get_refractive_indices(self, wavelengths):
        """
            Interpolate the refractive indices at the desired wavelengths [in nm]
            We use a two-parameter model n(lambda) = A + B / lambda**2
            A and B are recovered from the refractive index at the "d" wavelength and the Abbe number
            See "End-to-End Complex Lens Design with Differentiable Ray Tracing" (Sun et al, 2021)
        """
        wc = 656.3
        wd = 587.6
        wf = 486.1
        b = (self.nd - 1) / (self.v * (wf ** -2 - wc ** -2))
        a = self.nd - b / wd ** 2
        n = a[..., None] + b[..., None] / np.array([[wavelengths]]) ** 2
        n = tf.where(self.structure.mask_G[..., None], n, tf.ones_like(n))
        return n

    @property
    def efl(self):
        return rt.get_first_order(self)[0]

    @property
    def bfl(self):
        return rt.get_first_order(self)[1]

    @property
    def entrance_pupil_position(self):
        return rt.compute_pupil_position(self)
