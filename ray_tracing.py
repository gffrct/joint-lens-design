"""
    Classes and functions that make use of optical ray tracing

    In general, the tensors are arranged as follows:
        Dim 0: Number of samples (optical systems)
        Dim 1: Number of field angles
        Dim 2: Number of pupil intersections
        Dim 3: Number of wavelengths
        Dim 4: Number of surfaces
"""

# Tensorflow
import tensorflow as tf

# Others
import numpy as np


def mask_replace(mask, src, dst):
    assert src.shape == mask.shape
    assert len(dst.shape) == 1
    return tf.where(mask, tf.scatter_nd(tf.where(mask), dst, mask.shape), src)


class RayTracer:

    def __init__(self, mode='skew_random', n_rays=(8, 8), rel_fields=(0., 0.707, 1.), vig_fn=None,
                 double_precision=False, wavelengths=(656.3, 587.6, 486.1), n_ray_aiming_iter=0,
                 ray_aiming_mode='real', allow_backward_rays=True):
        self.mode = mode

        if self.mode == 'skew_random':
            assert len(n_rays) == 2
            self.pupil_span = lambda tensor: circle_pseudo_random(tensor, *n_rays)
        elif self.mode == 'skew_uniform_half_equidistant':
            assert len(n_rays) == 2
            self.pupil_span = lambda tensor: skew_uniform_half_equidistant(tensor, *n_rays)
        elif self.mode == 'skew_uniform_half_jittered':
            assert len(n_rays) == 2
            self.pupil_span = lambda tensor: skew_uniform_half_jittered(tensor, *n_rays)
        elif self.mode == 'skew_inner_square_half':
            self.pupil_span = lambda tensor: skew_inner_square_half(tensor, *n_rays)
        elif self.mode == 'skew_outer_edge_uniform':
            self.pupil_span = lambda tensor: circle_outer_edge_uniform(tensor, n_rays)
        elif self.mode == 'meridional_uniform':
            self.pupil_span = lambda tensor: meridional_uniform(tensor, n_rays)
        elif self.mode == 'sagittal_uniform':
            self.pupil_span = lambda tensor: sagittal_uniform(tensor, n_rays)
        elif self.mode == 'chief':
            self.pupil_span = lambda tensor: chief(tensor, n_rays)
        elif self.mode == 'tee':
            self.pupil_span = lambda tensor: tee(tensor)
        else:
            assert ValueError('Ray tracing mode must be either "skew_random", "skew_outer_edge_uniform", '
                              '"meridional_uniform", "sagittal_uniform", "tee" or "chief"')

        self.n_rays = n_rays
        self.rel_fields = rel_fields
        self.vig_fn = vig_fn
        self.n_ray_aiming_iter = n_ray_aiming_iter
        self.ray_aiming_mode = ray_aiming_mode
        self.allow_backward_rays = allow_backward_rays

        # Wavelengths
        self.wavelengths = wavelengths
        conversion = {
            'C': 656.3,
            'd': 587.6,
            'F': 486.1
        }
        self.wavelengths = [conversion[w] if w in conversion.keys() else w for w in wavelengths]

        self.double_precision = double_precision

    def trace_rays(self, specs, lens, use_vig=True, aggregate=False, xy=None, up_to_stop=False):

        if self.double_precision:
            specs = specs.double()
            lens = lens.double()

        n = lens.get_refractive_indices(self.wavelengths)
        n = tf.concat((tf.ones_like(n[:, 0:1, :]), n), axis=1)
        n = tf.transpose(n, perm=[0, 2, 1])
        n = tf.reshape(n, (n.shape[0], 1, 1, n.shape[1], -1))

        z = tf.reshape(compute_pupil_position(lens), (-1, 1, 1, 1))

        # Find x and y coordinates at the entrance pupil
        if xy is None:
            xp_rel, yp_rel = self.pupil_span(z)
        else:
            xp_rel, yp_rel = xy
        if use_vig and self.vig_fn is not None and self.mode != 'chief':
            fields = tf.constant(self.rel_fields, dtype=tf.float32)[None, :]
            vig_up = self.vig_fn(fields, specs.vig_up)
            vig_down = self.vig_fn(fields, specs.vig_down)
            vig_x = self.vig_fn(fields, specs.vig_x)
            yp_rel = apply_vignetting(yp_rel, vig_up, vig_down)
            xp_rel = apply_vignetting(xp_rel, vig_x, vig_x)

        # Apply ray aiming to correct the 'x' and 'y' intersections at the pupil
        if self.n_ray_aiming_iter > 0 and up_to_stop is False:
            ray_aiming_fn = self.ray_aiming(specs, lens.detach(), use_vig)
            xp_rel, yp_rel = [tf.stop_gradient(tf.clip_by_value(item, -2, 2)) for item in ray_aiming_fn(xp_rel, yp_rel)]

        xp = scale_to_epd(xp_rel, specs.epd)
        yp = scale_to_epd(yp_rel, specs.epd)

        # Find the direction cosines
        u = (specs.hfov[:, None] * tf.constant(self.rel_fields, dtype=tf.float32)[None, :])[..., None, None]
        cy = tf.sin(u)
        cx = tf.reshape(tf.zeros(1), (1, 1, 1, 1))

        # Adjust dimensions
        c = tf.reshape(lens.c, (lens.c.shape[0], 1, 1, 1, -1))
        t = tf.reshape(lens.t, (lens.t.shape[0], 1, 1, 1, -1))
        mu = n[..., :-1] / n[..., 1:]
        mask = tf.reshape(lens.structure.mask, (lens.c.shape[0], 1, 1, 1, -1))

        # Trace rays
        return trace_skew(xp, yp, z, cx, cy, c, t, mu, mask, aggregate, self.allow_backward_rays)

    def ray_aiming(self, specs, lens, use_vig):
        if (lens.structure.stop_idx == 0).all():
            # If the stop index is at the first position for all lenses, return the identity function
            return lambda xp_rel, yp_rel: (xp_rel, yp_rel)
        specs2stop = specs.up_to_stop()
        lens2stop = lens.up_to_stop()

        # Compute the stop radius (could be batched with the following ray tracing)
        if self.ray_aiming_mode == 'paraxial':
            magnification = compute_magnification(lens2stop)
            rs = tf.reshape(magnification * specs2stop.epd / 2, (-1, 1, 1, 1))
        elif self.ray_aiming_mode == 'real':
            rs = tf.reshape(compute_pupil_radius(specs2stop, lens2stop), (-1, 1, 1, 1))
        else:
            raise ValueError

        ray_aiming_fn = None

        # Generate lower and upper meridional rays as well as a sagittal ray for all lenses, fields, and colors
        xp_tee, yp_tee = tee(None)
        nw = len(self.wavelengths)
        shape = (len(lens), len(self.rel_fields), xp_tee.shape[2], nw)
        xp_tee = tf.broadcast_to(xp_tee, shape)
        yp_tee = tf.broadcast_to(yp_tee, shape)
        if use_vig and self.vig_fn:
            fields = tf.constant(self.rel_fields, dtype=tf.float32)[None, :]
            vig_down = self.vig_fn(fields, specs.vig_down)
            vig_up = self.vig_fn(fields, specs.vig_up)
            vig_x = self.vig_fn(fields, specs.vig_x)
            yp_tee = apply_vignetting(yp_tee, vig_up, vig_down)
            xp_tee = apply_vignetting(xp_tee, vig_x, vig_x)
        xp_tee_ref, yp_tee_ref = tf.identity(xp_tee), tf.identity(yp_tee)

        for k in range(self.n_ray_aiming_iter):
            if ray_aiming_fn:
                xp_tee, yp_tee = ray_aiming_fn(xp_tee, yp_tee)

            # Trace those rays up to the aperture stop
            # Subscript 's' means 'aperture stop'; subscript 'p' means 'entrance pupil'
            with tf.GradientTape() as tape:
                tape.watch(xp_tee)
                tape.watch(yp_tee)
                xs, ys, *_ = self.trace_rays(specs2stop, lens2stop, up_to_stop=True, use_vig=False, xy=(xp_tee, yp_tee))

                # Find the intersections in relative units
                xs_rel = xs / rs
                ys_rel = ys / rs

            # Compute the gradient of the relative pupil coordinates w.r.t. the relative entrance pupil coordinates
            x_grad, y_grad = tape.gradient((xs_rel, ys_rel), (xp_tee, yp_tee))

            # Compute the error between the aperture stop coordinates and entrance pupil coordinates
            delta_xs_tee = xs_rel - xp_tee_ref
            delta_ys_tee = ys_rel - yp_tee_ref

            # Compute the relative correction factors in pupil space
            delta_xp_tee = - delta_xs_tee / x_grad
            delta_yp_tee = - delta_ys_tee / y_grad

            # Solve numerical stability issues
            # This is equivalent to disabling ray aiming
            delta_xp_tee = tf.where(tf.math.is_finite(delta_xp_tee), delta_xp_tee, 0)
            delta_yp_tee = tf.where(tf.math.is_finite(delta_yp_tee), delta_yp_tee, 0)

            # Define a function to linearly interpolate between the relative correction factors
            delta_xp = delta_xp_tee[..., -1:, :]
            delta_yp_l, delta_yp_u = tf.split(delta_yp_tee[..., :2, :], 2, axis=2)
            xp = xp_tee[..., -1:, :]
            yp_l, yp_u = tf.split(yp_tee[..., :2, :], 2, axis=2)
            yp_scale = (yp_u + delta_yp_u - (yp_l + delta_yp_l)) / (yp_u - yp_l)
            yp_offset = (yp_l * delta_yp_u - yp_u * delta_yp_l) / (yp_l - yp_u)

            # Update the coordinates
            def ray_aiming_fn(xp_rel, yp_rel):
                return xp_rel * (xp + delta_xp) / xp, yp_rel * yp_scale + yp_offset

        return ray_aiming_fn


def compute_psf(x, y, n_bins=(21, 21), increment=None, y_target=None):
    """
        x, y: shape [n_lens, n_fields, n_channels, n_rays]
    """
    nw = x.shape[-2]

    # One grid for every field of every system
    n_grids = x.shape[0] * x.shape[1]

    # Compute the grid dimensions and centroids ('x' centroid is always 0)
    n_x_bins, n_y_bins = n_bins
    if y_target is None:
        y_target = tf.reduce_mean(tf.reshape(y, (n_grids, -1)), axis=1)

    # Center the y coordinates at y_target
    y = y - y_target[:, None, None]

    if increment is not None:
        x_incr = y_incr = tf.ones(n_grids) * increment
        x_size = increment * n_x_bins
        y_size = increment * n_x_bins

    else:
        y_min = tf.reduce_min(tf.reshape(y, (n_grids, -1)), axis=1)
        y_max = tf.reduce_max(tf.reshape(y, (n_grids, -1)), axis=1)
        x_size = tf.reduce_max(tf.reshape(x, (n_grids, -1)), axis=1)
        y_size = 2 * tf.maximum(y_max - y_target, y_target - y_min)

        x_incr = x_size / n_x_bins
        y_incr = y_size / n_y_bins

    # Compute the pixel center coordinates
    # Use symmetry to compute only half of 'x' bins
    if n_x_bins % 2 == 1:
        grid_coords_x = (tf.range(n_x_bins // 2 + 1, dtype=tf.float32))[None, :] * x_incr[:, None]
    else:
        grid_coords_x = (tf.range(n_x_bins // 2, dtype=tf.float32) + 0.5)[None, :] * x_incr[:, None]
    grid_coords_y = (tf.range(n_y_bins, dtype=tf.float32) + 0.5 - n_y_bins / 2)[None, :] * y_incr[:, None]

    # Compute the soft histogram
    sigma_x = x_incr / 2
    sigma_y = y_incr / 2
    dist_x_squared = (tf.reshape(x, (n_grids, nw, 1, 1, -1)) -
                      tf.reshape(grid_coords_x, (n_grids, 1, 1, -1, 1))) ** 2
    dist_y_squared = (tf.reshape(y, (n_grids, nw, 1, 1, -1)) -
                      tf.reshape(grid_coords_y, (n_grids, 1, -1, 1, 1))) ** 2
    gaussian_x = tf.exp(- (dist_x_squared / tf.reshape(sigma_x, (-1, 1, 1, 1, 1)) ** 2) / 2)
    gaussian_y = tf.exp(- (dist_y_squared / tf.reshape(sigma_y, (-1, 1, 1, 1, 1)) ** 2) / 2)
    gaussian = gaussian_x * gaussian_y
    kernels = tf.reduce_sum(gaussian, axis=-1)

    # Get the negative 'x' values
    if n_x_bins % 2 == 1:
        kernels = tf.concat((tf.reverse(kernels[..., 1:], axis=(-1,)), kernels), axis=-1)
    else:
        kernels = tf.concat((tf.reverse(kernels, axis=(-1,)), kernels), axis=-1)

    # Normalize to have unit area
    kernels = kernels / tf.reduce_sum(kernels, axis=(-1, -2), keepdims=True)

    # Log the proportion of rays that hit the grid at different field values
    accounted_rays = (tf.abs(y) < y_size / 2) & (tf.abs(x) < x_size / 2)
    accounted_ray_proportion = tf.reduce_mean(tf.cast(accounted_rays, tf.float32), axis=(-1, -2))

    return x_size, y_size, y_target, kernels, accounted_ray_proportion


def compute_n(nd, v, glass_mask):
    """
        Compute the refractive indices for C, d and F wavelengths, respectively

        We assume linear partial dispersion P_{F, d} w.r.t. Abbe number
        We compute the linear model anchored by K7 and F2 glasses
    """
    alpha = -4.5757e-4
    beta = 7.2264e-1

    nf = nd + (nd - 1) * (alpha + beta / v)
    nc = nf - (nd - 1) / v

    mask = tf.repeat(tf.concat((tf.zeros_like(glass_mask[:, 0:1]), glass_mask), axis=1)[None, ...], 3, axis=0)

    n = tf.stack((nc, nd, nf), axis=0)
    n2d = tf.ones(mask.shape)

    n2d = mask_replace(mask, n2d, tf.reshape(n, (-1,)))
    n2d = tf.transpose(n2d, perm=[1, 0, 2])
    return n2d


def reduce_abcd(abcd):
    """
        Reduce the ABCD matrices using a recurrent algorithm to reduce the kernel launches
    """
    while abcd.shape[1] > 1:
        if abcd.shape[1] % 2 == 0:
            abcd = abcd[:, 1::2, ...] @ abcd[:, ::2, ...]
        else:
            abcd = tf.concat((abcd[:, 1::2, ...] @ abcd[:, :-1:2, ...], abcd[:, -1:, ...]), axis=1)

    return tf.squeeze(abcd, axis=1)


def interface_propagation_abcd(c, t, n):
    """
        Batch the computation of the ABCD matrix of a spherical interface followed by a propagation
    """
    assert n.shape[-1] - 1 == c.shape[-1] == t.shape[-1]

    D = n[:, :-1] / n[:, 1:]  # D = n / n_prime
    C = c * (D - 1)  # C = c * (n - n_prime) / n_prime
    A = 1 + C * t  # A = 1 + t * c * (n - n_prime) / n_prime
    B = D * t  # B = t * n / n_prime

    abcd = tf.reshape(tf.stack((A, B, C, D), axis=-1), (n.shape[0], -1, 2, 2))

    return abcd


def compute_pupil_position(lens):
    """
        Compute the position of the paraxial entrance pupil w.r.t. the first optical surface

        We compute the ABCD matrix of all components previous to the aperture stop
        Then the pupil location w.r.t. the first surface is given by the ratio B/A
    """
    # Get the lens up to the aperture stop
    lens = lens.up_to_stop()
    if lens.structure.mask.shape[1] != 0:
        nd = tf.concat((tf.ones_like(lens.nd[:, 0:1]), lens.nd), axis=1)

        # Compute the ABCD matrix
        all_abcd = interface_propagation_abcd(lens.c, lens.t, nd)
        abcd = reduce_abcd(all_abcd)

        pupil_position = abcd[:, 0, 1] / abcd[:, 0, 0]
    else:
        pupil_position = tf.zeros(len(lens))

    return pupil_position


def tee(tensor):
    """
        Compute bottom meridional ray, top meridional ray, and positive sagittal ray
    """
    y = tf.reshape(tf.constant([-1., 1., 0.]), (1, 1, -1, 1))
    x = tf.reshape(tf.constant([0., 0., 1.]), (1, 1, -1, 1))

    return x, y


def meridional_uniform(tensor, n_rays):
    """
        Compute 'n_rays' x and y relative meridional pupil intersections to span the pupil uniformly
    """
    y = tf.reshape(tf.linspace(-1., 1., n_rays), (1, 1, -1, 1))
    x = tf.zeros_like(y)

    return x, y


def sagittal_uniform(tensor, n_rays):
    """
        Compute 'n_rays' x and y relative positive sagittal pupil intersections to span the pupil uniformly
    """
    x = tf.linspace(0., 1., n_rays)[None, None, :, None]
    y = tf.zeros_like(x)

    return x, y


def chief(tensor, _):
    """
        Output the relative position of the chief ray
    """
    x = tf.zeros((1, 1, 1, 1))
    y = tf.zeros((1, 1, 1, 1))

    return x, y


def circle_pseudo_random(tensor, n_r, n_theta):
    """
        Compute 'n_r' * 'n_theta' x and y relative pupil intersections to span the pupil uniformly and randomly
        The rays are broadcasted to 'tensor' dimensions
    """
    n_rays = n_r * n_theta
    n_elements = tf.reduce_prod(tensor.shape)
    delta_r_squared = (tf.random.uniform((n_elements, n_r, n_theta)) / n_r)
    delta_theta = (tf.random.uniform((n_elements, n_r, n_theta)) / n_theta)
    r_squared_increments = tf.constant(np.linspace(0, 1, n_r, endpoint=False, dtype=np.float32))[None, :, None]
    theta_increments = tf.constant(np.linspace(0, 1, n_theta, endpoint=False, dtype=np.float32))[None, None, :]
    r_squared = delta_r_squared + r_squared_increments
    theta = (delta_theta + theta_increments) * 2 * np.pi
    r = tf.sqrt(r_squared)
    x = r * tf.cos(theta)
    y = r * tf.sin(theta)

    return tf.reshape(x, (-1, 1, n_rays, 1)), tf.reshape(y, (-1, 1, n_rays, 1))


def skew_uniform_half_equidistant(tensor, n_r, n_i):
    """
        Compute (n_r ** 2)(n_i) x and y relative pupil intersections to span the right half of the pupil uniformly
    """
    rays_per_shell = [n_i * (i * 2 + 1) for i in range(n_r)]
    shell_idx = [j for i in range(n_r) for j in ([i] * n_i * (i * 2 + 1))]
    r = ((np.arange(n_r) + 0.5) / n_r)[shell_idx]
    theta = [(i / n - 0.5) * np.pi for n in rays_per_shell for i in (np.arange(n) + 0.5)]

    x = tf.constant(r * np.cos(theta), dtype=tf.float32)
    y = tf.constant(r * np.sin(theta), dtype=tf.float32)

    return tf.reshape(x, (1, 1, -1, 1)), tf.reshape(y, (1, 1, -1, 1))


def skew_uniform_half_jittered(tensor, n_r, n_i):
    """
        Compute (n_r ** 2)(n_i) x and y relative pupil intersections to span the right half of the pupil uniformly
        The sampling pattern is somewhat biased but samples the outer edge of the pupil
    """
    rays_per_shell = np.array([n_i * (i * 2 + 1) for i in range(n_r)])
    shell_idx = np.array([j for i in range(n_r) for j in ([i] * n_i * (i * 2 + 1))])
    inner_r = np.linspace(0, 1, n_r * 2)[::2]
    delta_r = 1 / (2 * n_r - 1)
    r = inner_r[shell_idx] + delta_r * ((np.arange(len(shell_idx)) + np.array(shell_idx)) % 2)
    theta = [(i / n - 0.5) * np.pi for n in rays_per_shell for i in (np.arange(n) + 0.5)]

    x = tf.constant(r * np.cos(theta), dtype=tf.float32)
    y = tf.constant(r * np.sin(theta), dtype=tf.float32)

    return tf.reshape(x, (1, 1, -1, 1)), tf.reshape(y, (1, 1, -1, 1))


def skew_inner_square_half(tensor, n_y, _):
    """
        Compute n_x * n_y relative pupil intersections to span the pupil following an inner square
    """
    x = np.linspace(-1, 1, n_y * 2)[-n_y:] / np.sqrt(2)
    y = np.linspace(-1, 1, n_y) / np.sqrt(2)
    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)
    x = x[None, :] * tf.ones_like(y[:, None])
    y = y[:, None] * tf.ones_like(x[None, :])

    return tf.reshape(x, (1, 1, -1, 1)), tf.reshape(y, (1, 1, -1, 1))


def circle_outer_edge_uniform(tensor, n_rays):
    """
        Compute 'n_rays' x and y relative pupil intersections to span the outer edge of the pupil uniformly
    """
    theta = tf.constant(np.linspace(0, 2 * np.pi, n_rays, endpoint=False, dtype=np.float32))
    x = tf.cos(theta)
    y = tf.sin(theta)

    return tf.reshape(x, (1, 1, -1, 1)), tf.reshape(y, (1, 1, -1, 1))


def apply_vignetting(y, vig_up, vig_down):
    """
        Apply vignetting in the y dimension to the normalized pupil coordinates
    """
    trailing_dims = [1] * (len(y.shape) - len(vig_down.shape))
    vig_up = tf.reshape(vig_up, (*vig_up.shape, *trailing_dims))
    vig_down = tf.reshape(vig_down, (*vig_down.shape, *trailing_dims))
    scale = 1 - (vig_up + vig_down) / 2
    offset = (vig_down - vig_up) / 2
    y = y * scale + offset

    return y


def scale_to_epd(y, epd):
    """
        Given the entrance pupil position,
        compute the ray heights and angles at the first optical surface

        We assume infinite conjugates
    """
    trailing_dims = [1] * (len(y.shape) - 1)
    y = y * tf.reshape(epd, (-1, *trailing_dims)) / 2

    return y


def sin2cos(sin):
    return tf.sqrt(1 - sin ** 2)


def update_ray_coordinates(x, y, z, cx, cy, cz, distance):
    """
        Update the ray position vectors from the ray marching distance and direction cosines
    """
    delta_z = distance * cz
    x = x + distance * cx
    y = y + distance * cy
    z = z + delta_z
    return x, y, z, delta_z


def find_marching_distance_spherical(c, x, y, z, cx, cy, cz):
    """
        Find the ray marching distance required to reach the spherical surface
        Return intermediate values as well
    """
    eps = 1e-6

    e = - (x * cx + y * cy + z * cz)
    mz = z + e * cz
    m2 = x ** 2 + y ** 2 + z ** 2 - e ** 2
    temp = c * m2 - 2 * mz
    cos2_theta = cz ** 2 - c * temp

    # Check for missed rays
    # Allow cos(theta)^2 to be above 1 due to numerical errors, but not below "eps"
    failures = cos2_theta - eps < 0

    cos_theta = tf.sqrt(tf.where(~failures, cos2_theta, 1))
    dist = e + temp / (cz + cos_theta)

    return failures, dist, cos_theta, cos2_theta


def apply_snell_spherical(c, mu, x, y, cx, cy, cos_theta):
    """
        Update the direction cosines after refraction at the interface
    """
    eps = 1e-6

    cos2_prime = 1 - mu ** 2 * (1 - cos_theta ** 2)

    # Check for total internal reflexion
    # Allow cos(theta')^2 to be above 1 due to numerical errors, but not below "eps"
    failures = cos2_prime - eps < 0

    cos_prime = tf.sqrt(tf.where(~failures, cos2_prime, 1))
    g = cos_prime - mu * cos_theta
    cx = mu * cx - g * c * x
    cy = mu * cy - g * c * y
    cz2 = 1 - (cx ** 2 + cy ** 2)

    # Check for numerical failures
    failures = failures | (cz2 - eps < 0)
    cz = tf.sqrt(tf.where(~failures, cz2, 1))
    # Alternative: cz = mu * cz - g * (c * z - 1)

    return failures, cx, cy, cz, cos2_prime


def reset_bad_rays(ray_ok, x, y, z, cx, cy, cz, normalize=False):
    """
        Reset the position and direction vectors of rays that didn't trace successfully
        The goal is to avoid NaNs in the forward/backward pass
        If "normalize", re-normalize the direction vectors to prevent the propagation of numerical errors
    """
    x = tf.where(ray_ok, x, 0)
    y = tf.where(ray_ok, y, 0)
    z = tf.where(ray_ok, z, 0)
    cx = tf.where(ray_ok, cx, 0)
    cy = tf.where(ray_ok, cy, 0)
    cz = tf.where(ray_ok, cz, 1)
    if normalize:
        norm = (cx ** 2 + cy ** 2 + cz ** 2).sqrt()
        cx = cx / norm
        cy = cy / norm
        cz = cz / norm
    return x, y, z, cx, cy, cz


def trace_skew(x, y, z, cx, cy, c, t, mu, mask, aggregate=False, allow_backward_rays=True):
    """
        Given x, y and z as well as the direction cosines cx and cy at the entrance pupil,
        compute the ray intersections at the image plane
    """
    stacks = {k: [] for k in ('x', 'y', 'z', 'cos2', 'cos2_prime')}

    c = tf.unstack(c, axis=-1)
    t = tf.unstack(t, axis=-1)
    mu = tf.unstack(mu, axis=-1)
    mask = tf.unstack(mask, axis=-1)

    ray_ok = tf.ones_like(y, dtype=tf.bool)
    ray_backward = tf.zeros_like(y, dtype=tf.bool)

    cz = tf.sqrt(1 - cx ** 2 - cy ** 2)

    for k in range(len(t)):
        # Find ray marching distance
        failures, distance, cos_theta, cos2_theta = find_marching_distance_spherical(c[k], x, y, z, cx, cy, cz)

        # Update ray coordinates
        x, y, z, delta_z = update_ray_coordinates(x, y, z, cx, cy, cz, distance)

        # Check for ray failures, update the penalty, and reset
        ray_ok = ray_ok & ~failures
        x, y, z, cx, cy, cz = reset_bad_rays(ray_ok, x, y, z, cx, cy, cz, normalize=False)

        # Apply Snell's law and update direction cosines
        failures, cx, cy, cz, cos2_prime = apply_snell_spherical(c[k], mu[k], x, y, cx, cy, cos_theta)

        # Penalize rays that travel backward except for the ones coming from the entrance pupil
        if k > 0:
            # Don't take into account rays that failed or rays going through dummy surfaces
            mask_k = ray_ok & mask[k - 1]
            if allow_backward_rays:
                ray_backward = ray_backward | ((delta_z < 0) & mask_k)
            else:
                ray_ok = ray_ok & ~((delta_z < 0) & mask_k)

        # Check for ray failures, update the penalty, and reset
        ray_ok = ray_ok & ~failures
        x, y, z, cx, cy, cz = reset_bad_rays(ray_ok, x, y, z, cx, cy, cz, normalize=False)

        # Center coordinate system at vertex of next surface
        z = z - t[k]

        if aggregate:
            stacks['x'].append(tf.broadcast_to(x, (*x.shape[:3], mu[0].shape[-1])))
            stacks['y'].append(tf.broadcast_to(y, (*y.shape[:3], mu[0].shape[-1])))
            stacks['z'].append(tf.broadcast_to(z, (*z.shape[:3], mu[0].shape[-1])))
            stacks['cos2'].append(tf.broadcast_to(cos2_theta, (*x.shape[:3], mu[0].shape[-1])))
            stacks['cos2_prime'].append(tf.broadcast_to(cos2_prime, (*x.shape[:3], mu[0].shape[-1])))

    # Transfer to image plane
    delta_z = - z
    dist = delta_z / cz
    x = x + dist * cx
    y = y + dist * cy

    # Penalize rays that travel backward
    mask_k = ray_ok & mask[-1]
    if allow_backward_rays:
        ray_backward = ray_backward | ((delta_z < 0) & mask_k)
    else:
        ray_ok = ray_ok & ~((delta_z < 0) & mask_k)

    if aggregate:
        stacks['x'].append(x)
        stacks['y'].append(y)
        stacks['z'].append(z + delta_z)
        return x, y, cx, cy, ray_ok, ray_backward, stacks

    return x, y, cx, cy, ray_ok, ray_backward


def compute_rms2d(x, y, ray_ok):
    """
        Compute the mean rms spot size for every sample

        Coordinates of rays that fail to pass the system (~ray_ok) are remapped to the mean values
    """
    eps = 1e-6
    ray_ok = tf.cast(ray_ok, tf.float32)

    # Fill invalid data with zeros
    y = y * ray_ok
    x = x * ray_ok

    # Compute the mean of rays from every field angle from valid data only
    # Mean x is 0 for rotationally symmetric systems
    y_mean = tf.reduce_sum(y, axis=(2, 3), keepdims=True) / (tf.reduce_sum(ray_ok, axis=(2, 3), keepdims=True) + eps)

    # Fill invalid data with the mean ray height
    y = y + (1 - ray_ok) * y_mean

    x_var = tf.reduce_mean(x ** 2, axis=(2, 3))
    y_var = tf.reduce_mean((y - y_mean) ** 2, axis=(2, 3))
    rms = tf.sqrt(tf.maximum(x_var + y_var, eps ** 2))
    return rms


def compute_last_curvature(structures, c, t, nd):
    """
        Compute the last curvature of the system so that the effective focal length is 1
    """
    mask = structures.mask
    seq_length = mask.sum(axis=1)
    # Detect if the last two elements are both air
    indices = tf.stack((tf.range(mask.shape[0]), seq_length - 2), axis=1)
    air_air = ~tf.gather_nd(structures.mask_G, indices)
    # Find the index of the last curvature between a glass element and an air gap
    last_c_idx = seq_length - 1 - tf.cast(air_air, tf.int32)
    # Do not use the elements with respect to the last surface for the computation
    indices = tf.stack((tf.range(mask.shape[0]), seq_length - 1), axis=1)
    c_mask = mask & ~tf.cast(tf.scatter_nd(indices, tf.constant([1.] * indices.shape[0]), mask.shape), tf.bool)

    c2d = mask_replace(c_mask, tf.zeros_like(mask, dtype=tf.float32), c)
    t2d = mask_replace(mask, tf.zeros_like(mask, dtype=tf.float32), t)
    n2d = mask_replace(structures.mask_G, tf.ones_like(mask, dtype=tf.float32), nd)
    n2d = tf.concat((tf.ones_like(n2d[:, 0:1]), n2d), axis=1)

    # For sequences that end with air-air, do not use the last element either
    indices = tf.stack((tf.range(mask.shape[0]), last_c_idx), axis=1)
    selection_mask = c_mask & ~tf.cast(tf.scatter_nd(indices, tf.constant([1.] * indices.shape[0]), mask.shape),
                                       tf.bool)

    abcd = interface_propagation_abcd(c2d, t2d, n2d)
    identity_matrix = tf.eye(2)[None, None, ...]
    abcd = tf.where(tf.broadcast_to(selection_mask[..., None, None], abcd.shape), abcd,
                    tf.broadcast_to(identity_matrix, abcd.shape))
    abcd = reduce_abcd(abcd)

    # We assume that the image plane is in air
    # The last curvature c is computed as: c = - (1 + n * C) / (A * (n - 1))
    # where A and C are elements of the ABCD matrix
    # and where n is the refractive index before the last interface
    indices = tf.stack((tf.range(mask.shape[0]), last_c_idx), axis=1)
    last_n = tf.gather_nd(n2d, indices)
    last_c = - (1 + last_n * abcd[:, 1, 0]) / (abcd[:, 0, 0] * (last_n - 1))

    indices = tf.stack((tf.range(mask.shape[0]), last_c_idx), axis=1)
    c2d = tf.tensor_scatter_nd_update(c2d, indices, last_c)

    return c2d[mask]


def get_first_order(lens):
    """
        Compute both the EFL and BFL of the lens
    """
    nd = tf.concat((tf.ones_like(lens.nd[:, 0:1]), lens.nd), axis=1)
    t = lens.t

    # Get the ABCD matrix of the system excluding the image plane
    indices = tf.stack((tf.range(len(lens)), np.sum(lens.structure.mask, axis=1) - 1), axis=1)
    # Zero-out the last thickness
    t = tf.tensor_scatter_nd_update(t, indices, tf.constant([0.] * indices.shape[0]))
    abcd = interface_propagation_abcd(lens.c, t, nd)
    abcd = reduce_abcd(abcd)

    # Effective focal length: EFL = - 1 / C
    efl = - 1 / abcd[:, 1, 0]

    # Back focal length: BFL = - A / C
    bfl = - abcd[:, 0, 0] / abcd[:, 1, 0]

    return efl, bfl


def compute_magnification(lens):
    """
        Compute the first-order magnification of the lens
    """
    # Get the ABCD matrix
    nd = tf.concat((tf.ones_like(lens.nd[:, 0:1]), lens.nd), axis=1)
    abcd = interface_propagation_abcd(lens.c, lens.t, nd)
    abcd = reduce_abcd(abcd)

    # The magnification corresponds to the A element
    magnification = abcd[:, 0, 0]

    return magnification


def get_paraxial_heights_at_image_plane(specs, lens, relative_fields):
    """
        Compute the paraxial heights at the image plane

        We consider the height of a paraxial chief ray (that hits the entrance pupil at the middle)
    """
    angles = tf.constant(relative_fields, dtype=tf.float32)[None, :] * specs.hfov[:, None]

    # Compute the derivative of the chief ray height as a function of the field angle
    # This corresponds to B' = B - A * pupil_position
    pupil_position = compute_pupil_position(lens)
    nd = tf.concat((tf.ones_like(lens.nd[:, 0:1]), lens.nd), axis=1)
    abcd = reduce_abcd(interface_propagation_abcd(lens.c, lens.t, nd))
    a, b = abcd[:, 0, 0], abcd[:, 0, 1]
    b_prime = b - a * pupil_position

    # Compute the reference heights, knowing that they are proportional to the tangent of the field angle
    heights = tf.tan(angles) * b_prime

    return heights


def compute_pupil_radius(specs, lens2stop):
    """
        Compute the pupil radius using a marginal ray up to the aperture stop
    """
    x = tf.zeros([1, 1, 1, 1])
    y = tf.ones([1, 1, 1, 1])

    tracer = RayTracer(rel_fields=[0.], vig_fn=None, wavelengths=['d'])
    xp, yp, *_ = tracer.trace_rays(specs, lens2stop, xy=(x, y), use_vig=False)

    return tf.squeeze(yp, axis=(1, 2, 3))


def compute_distortion(specs, lens, relative_fields):
    # Do ray tracing with chief ray
    # Vignetting is not required
    _, y, _, cy, *_ = RayTracer(mode='chief', rel_fields=relative_fields, wavelengths=['d'],
                                vig_fn=None).trace_rays(specs, lens)
    y, cy = tf.reshape(y, (len(specs), -1)), tf.reshape(cy, (len(specs), -1))

    # Compute paraxial height at paraxial image plane (PIM)
    relative_fields = tf.constant(relative_fields)
    paraxial_heights = tf.tan(relative_fields[None, :] * specs.hfov[:, None]) * lens.efl[:, None]

    # Apply a correction to the paraxial height for the defocus
    # The defocus is the distance from the PIM to the image plane
    indices = tf.stack(
        (tf.range(len(specs)), tf.constant(specs.structure.mask.sum(axis=1) - 1, dtype=tf.int32)), axis=1)
    last_t = tf.gather_nd(lens.t, indices)
    defocus = last_t - lens.bfl
    ref_y = paraxial_heights + defocus[:, None] * cy / sin2cos(cy)

    # Compute the distortion
    distortion = (y - ref_y) / ref_y

    return distortion


def compute_relative_illumination(specs, lens, relative_fields, vig_fn, n_ray_aiming_iter=1, wavelengths=('d',)):
    """
        Estimation of relative illumination. First field must be 0.

        For every field, we trace two marginal rays and one sagittal ray. See https://doi.org/10.1117/12.938414
    """
    eps = 1e-6
    assert relative_fields[0] == 0.

    ray_tracer = RayTracer(rel_fields=relative_fields, vig_fn=vig_fn, n_ray_aiming_iter=n_ray_aiming_iter,
                           wavelengths=wavelengths)
    x = tf.reshape(tf.constant([0, 0, 1], dtype=tf.float32), (1, 1, -1, 1))
    y = tf.reshape(tf.constant([1, -1, 0], dtype=tf.float32), (1, 1, -1, 1))
    _, _, cx, cy, ray_ok, _ = ray_tracer.trace_rays(specs, lens, xy=(x, y))

    relative_illumination = (cy[..., 0, :] - cy[..., 1, :]) * cx[..., 2, :] / tf.maximum((2 * cy[:, 0, 0, 0] ** 2), eps)

    # Replace values for fields where ray failures occurred
    validity_mask = tf.reduce_all(ray_ok, axis=(2, 3))[..., None]
    validity_mask = validity_mask & validity_mask[:, 0, :][:, None, :]
    relative_illumination = tf.where(validity_mask, relative_illumination, 1.)

    return relative_illumination


def compute_ray_aiming_error(specs, lens, rel_fields, vig_fn, n_ray_aiming_iter, ray_aiming_mode):
    """
        Generate upper and lower meridional rays, and compute the relative ray aiming error on each ray
        We assume no vignetting in 'x'
    """
    specs = specs.up_to_stop()
    lens = lens.up_to_stop()
    if (lens.structure.stop_idx == 0).all():
        return 0

    # Compute the stop radius (could be batched with the following ray tracing)
    if ray_aiming_mode == 'paraxial':
        magnification = compute_magnification(lens)
        rs = tf.reshape(magnification * specs.epd / 2, (-1, 1, 1, 1))
    elif ray_aiming_mode == 'real':
        rs = tf.reshape(compute_pupil_radius(specs, lens), (-1, 1, 1, 1))
    else:
        raise ValueError

    y = tf.reshape(tf.constant([-1., 1.]), (1, 1, -1, 1))
    x = tf.reshape(tf.constant([0., 0.]), (1, 1, -1, 1))

    tracer = RayTracer(rel_fields=rel_fields, vig_fn=vig_fn, wavelengths=['d'],
                       n_ray_aiming_iter=n_ray_aiming_iter, ray_aiming_mode=ray_aiming_mode)
    xp, yp, *_ = tracer.trace_rays(specs, lens, xy=(x, y), use_vig=True)

    if vig_fn is not None:
        fields = tf.constant(rel_fields, dtype=tf.float32)[None, :]
        vig_down = vig_fn(fields, specs.vig_down)
        vig_up = vig_fn(fields, specs.vig_up)
        # vig_x = vig_fn(fields, specs.vig_x)
        y = apply_vignetting(y, vig_up, vig_down)
        # x = apply_vignetting(x, vig_x, vig_x)

    error = (yp / rs - y)

    return error
