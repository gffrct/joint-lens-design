import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import yaml

import ray_tracing as rt
import lens_modeling as lm
import image_ops


class OpticsSimulator(tf.keras.Model):
    """
        Class to simulate optical aberrations on a given image
        The psfs, distortion shifts, and relative illumination factors
            need to be computed in children classes
            (e.g., with ray tracing or proxy model)
    """
    def __init__(self,
                 initial_lens_path,
                 add_bfl=True,
                 scale_factor=1,
                 detach=False,
                 trainable_vars=None,
                 disable_glass_optimization=False,
                 n_sampled_fields=21,
                 sensor_diagonal=16.,
                 psf_shape=(65, 65),
                 psf_abs_pixel_size=4.0e-3,
                 psf_grid_shape=(9, 9),
                 simulated_res_factor=1,
                 distortion_by_warping=True,
                 apply_distortion=True,
                 apply_relative_illumination=True,
                 lazy_init=False,
                 ):
        super(OpticsSimulator, self).__init__()

        if trainable_vars is None:
            self.trainable_vars = {'c': True, 't': True, 'g': True}
        elif isinstance(trainable_vars, bool):
            self.trainable_vars = {k: trainable_vars for k in ('c', 't', 'g')}
        else:
            self.trainable_vars = trainable_vars
        if disable_glass_optimization:
            self.trainable_vars['g'] = False
        self.detach = detach

        # Lens variables params
        if isinstance(initial_lens_path, dict):
            self.initial_lens = initial_lens_path
        else:
            with open(initial_lens_path, 'r') as f:
                # Load lens configuration and initial lens parameters
                self.initial_lens = yaml.safe_load(f)
        self.add_bfl = add_bfl
        self.scale_factor = scale_factor

        # SVOLA convolution params
        self.sensor_diagonal = sensor_diagonal
        self.n_fields = n_sampled_fields
        self.psf_shape = psf_shape
        self.psf_increment = psf_abs_pixel_size
        self.psf_grid_shape = psf_grid_shape
        # Optics model params
        self.simulated_res_factor = simulated_res_factor
        self.distortion_by_warping = distortion_by_warping
        self.apply_distortion = apply_distortion
        self.apply_relative_illumination = apply_relative_illumination
        # Metrics
        self.logged_metrics = {}

        self.lazy_init = lazy_init

    @classmethod
    def build_from_config(cls, config, **kwargs):
        keys = [k for k in config.DESCRIPTOR.fields_by_name if config.HasField(k) and k not in kwargs]
        kwargs_from_config = {k: getattr(config, k) for k in keys}
        return cls(**kwargs, **kwargs_from_config)

    def initialize(self):
        # Lens structure
        self.structure = lm.Structure(
            stop_idx=np.array(self.initial_lens['stop_idx']),
            sequence=np.array(self.initial_lens['sequence'])
        )

        # Lens specifications
        self.hfov = tf.Variable(initial_value=np.radians(np.array(self.initial_lens['hfov'])),
                                name='hfov', dtype=tf.float32,
                                trainable=False)
        self.f_number = tf.Variable(initial_value=np.array(self.initial_lens['f_number']),
                                    name='f_number', dtype=tf.float32,
                                    trainable=False)

        # Compute effective focal length required
        self.efl = self.sensor_diagonal / 2 / tf.tan(self.hfov)
        self.epd = self.efl / self.f_number
        self.specs = lm.Specs(self.structure, self.epd, self.hfov)

        # Lens variables - with normalized forms
        lens = lm.Lens(self.structure, *[tf.constant(self.initial_lens[key]) for key in ['c', 't', 'nd', 'v']])

        # The actual optimized variables are c, t and g
        self.c, self.t, self.g = lm.get_normalized_lens_variables(
            lens, self.trainable_vars, self.add_bfl, self.scale_factor)

        if not self.lazy_init:
            self.sample_optics_model()

    def get_vars(self):
        lens = self.lens
        return {
            'c_norm': self.c.numpy().tolist(),
            'g': self.g.numpy().tolist(),
            'nd': lens.nd.numpy().tolist(),
            'v': lens.v.numpy().tolist(),
            't': lens.t.numpy().tolist(),
            'lens_c': lens.c.numpy().tolist(),
            'stop_idx': self.structure.stop_idx.tolist(),
            'mask': lens.structure.mask.tolist(),
            'mask_G': lens.structure.mask_G.tolist(),
            'hfov': self.hfov.numpy().tolist(),
            'epd': self.epd.numpy().tolist(),
            'efl': self.efl.numpy().tolist(),
            'add_bfl': self.add_bfl,
        }

    def log_summary(self, step):
        for k, v in self.logged_metrics.items():
            tf.summary.scalar(k, v, step=step)
        psfs = self.sampled_psfs
        psfs = psfs / tf.reduce_max(psfs, axis=(1, 2, 3), keepdims=True)
        tf.summary.image('PSF', psfs, max_outputs=psfs.shape[0], step=step)

    @property
    def lens(self):
        """
            Recompute the lens from the normalized lens variables
        """
        lens = lm.get_lens_from_normalized(
            self.structure, self.c, self.t, self.g, self.catalog_g, self.add_bfl, self.scale_factor,
            self.quantized_continuous_glass_variables)
        # Scale the lens to the required EFL
        lens = lens.scale(self.efl / lens.efl)
        return lens

    def sample_roi_indices(self, roi_index=None):
        """
            For simulating higher resolutions, randomly sample a region of interest
                among "n_div ** 2" discrete choices
        """
        n_div = int(self.simulated_res_factor)
        if roi_index is None:
            roi_index = tf.random.uniform((1,), 0, n_div ** 2, dtype=tf.int32)[0]
        roi_index = roi_index % (n_div ** 2)
        row, col = tf.cast(roi_index // n_div, tf.float32), tf.cast(roi_index % n_div, tf.float32)
        return row, col

    def sample_field_lim(self, img_h, img_w, roi_index=None):
        """
            For higher simulated resolutions, compute the coordinates in object space
                that correspond to the image corners
            The coordinates are normalized such that x**2 + y**2 = 1
                corresponds to the outer edge of the circular full field of view
        """
        # Sample the ROI
        roi_indices = self.sample_roi_indices(roi_index=roi_index)
        row = roi_indices[0]
        col = roi_indices[1]
        # Retrieve the image limits in object-space coordinates
        factor = int(self.simulated_res_factor)
        diag = np.sqrt(img_h ** 2 + img_w ** 2)
        y0 = - img_h / diag * (2 * row / factor - 1)
        y1 = - img_h / diag * (2 * (row + 1) / factor - 1)
        x0 = img_w / diag * (2 * col / factor - 1)
        x1 = img_w / diag * (2 * (col + 1) / factor - 1)
        return x0, x1, y0, y1

    def compute_distortion_shift(self, x, y, x_lim, y_lim, field_lim):
        """
            Compute the shift of x and y due to distortion (relative to x_lim/y_lim)
            x_lim and y_lim represent the boundaries of the image coordinates
        """
        # Retrieve the relative field coordinates in object space
        # x_field ** 2 + y_field ** 2 = 1 correspond to the full field of view
        x0, x1, y0, y1 = field_lim
        x_field = (x - x_lim[0]) / (x_lim[1] - x_lim[0]) * (x1 - x0) + x0
        y_field = (y - y_lim[0]) / (y_lim[1] - y_lim[0]) * (y1 - y0) + y0

        # Interpolate the distortion shifts
        delta_x_field, delta_y_field = interpolate_distortion_shifts(
            self.sampled_distortion_shifts, x_field, y_field)

        # Scale the shifts w.r.t. the original image coordinates
        delta_x = delta_x_field * (x_lim[1] - x_lim[0]) / (x1 - x0)
        delta_y = delta_y_field * (y_lim[1] - y_lim[0]) / (y1 - y0)
        return delta_x, delta_y

    def sample_optics_model(self):
        return NotImplementedError

    def apply_optics_model(self, radiance, field_lim, max_value=255.):
        """
            Simulate the aberrations on the input radiance image
                according to the field limits
        """
        # Compute the "field map" to provide the relative field value for every pixel
        x0, x1, y0, y1 = field_lim
        img_h, img_w = radiance.shape[1:3]
        diag = np.sqrt(img_h ** 2 + img_w ** 2)
        y_map = tf.cast(tf.linspace(y0, y1, img_h), tf.float32)
        x_map = tf.cast(tf.linspace(x0, x1, img_w), tf.float32)
        field_map = tf.sqrt(x_map[None, :] ** 2 + y_map[:, None] ** 2)

        # Compute the required size for the PSFs in the PSF grid
        # Since the sensor and image aspect ratio don't necessarily match, we assume that the diagonal is the same
        psf_shape = np.array(self.psf_shape)
        factor = int(self.simulated_res_factor)
        resized_psf_shape = psf_shape * self.psf_increment * factor * diag / self.sensor_diagonal
        # Round to nearest odd number
        resized_psf_shape = (resized_psf_shape // 2 * 2 + 1).astype(np.int)

        # Interpolate PSFs, then rotate and resize
        psf_grid_shape = self.psf_grid_shape
        psfs = interpolate_psfs(self.sampled_psfs, field_map, psf_grid_shape)
        self.psfs = rotate_and_resize_psfs(psfs, x_map, y_map, psf_grid_shape, resized_psf_shape)

        # Apply SVOLA convolution
        overlap_size = (0.25 * (np.array(radiance.shape[1:3]) / self.psf_grid_shape)).astype(np.int)
        irradiance = image_ops.svola_convolution(radiance, overlap_size, self.psfs, self.psf_grid_shape, 'hann')

        psnr = tf.image.psnr(radiance, irradiance, max_value)
        ssim = tf.image.ssim(radiance, irradiance, max_value)

        if self.apply_relative_illumination:
            relative_illumination_map = interpolate_relative_illumination(self.sampled_relative_illumination, field_map)
            irradiance = irradiance * relative_illumination_map[None, ..., None]

        if self.apply_distortion and self.distortion_by_warping:
            # "field" are relative coordinates w.r.t. the field (object space)
            # "img" are relative coordinates w.r.t. to the image (-1 to 1)
            x_img = tf.broadcast_to(tf.cast(tf.linspace(-1., 1., img_w)[None, :], tf.float32), (img_h, img_w))
            x_img = tf.reshape(x_img, (-1,))
            y_img = tf.broadcast_to(tf.cast(tf.linspace(-1., 1., img_h)[:, None], tf.float32), (img_h, img_w))
            y_img = tf.reshape(y_img, (-1,))
            x_shift, y_shift = self.compute_distortion_shift(x_img, y_img, (-1, 1), (-1, 1), field_lim)
            # We subtract "x_shift" and "y_shift"
            x_img_dist = x_img - x_shift
            y_img_dist = y_img - y_shift
            irradiance = apply_distortion_by_warping(irradiance, x_img_dist, y_img_dist)

        return irradiance, psnr, ssim

    def call(self, radiance, training=None, mask=None, field_lim=None, recompute=True):
        if recompute:
            self.sample_optics_model()
            losses = self.get_losses().values()
            if len(losses) > 0:
                self.add_loss(tf.add_n(losses))

        if field_lim is None:
            # Simulate a higher resolution (by the given factor), and select a ROI randomly
            # We consider that the "radiance" image is the ROI
            field_lim = self.sample_field_lim(radiance.shape[1], radiance.shape[2])
        assert len(field_lim) == 4

        irradiance, psnr, ssim = self.apply_optics_model(radiance, field_lim)
        self.add_metric(psnr, aggregation='mean', name='IQ/psnr')
        self.add_metric(ssim, aggregation='mean', name='IQ/ssim')
        if self.detach:
            irradiance = tf.stop_gradient(irradiance)

        return irradiance


class RaytracedOptics(OpticsSimulator):
    """
        Class to simulate optical aberrations through exact ray tracing of a compound lens
        For convenience, the class also supports the optimization of the lens
            and computation of losses that act exclusively on the lens
    """
    def __init__(self,
                 initial_lens_path,
                 quantized_continuous_glass_variables=True,
                 wavelengths_r=(584.1, 604.2, 622.5, 642.2, 665.9),
                 wavelengths_g=(487.1, 512.1, 535.1, 560.8, 596.3),
                 wavelengths_b=(409.4, 435.4, 456.6, 477.9, 505.9),
                 n_pupil_rings=32,
                 n_ray_aiming_iter=1,
                 pupil_sampling='skew_uniform_half_jittered',
                 spot_size_weight=1,
                 ray_path_weight=100,
                 ray_path_lower_thresholds=(0.01, 1.0, 12.0),
                 ray_path_upper_thresholds=(None, 3.0, None),
                 ray_angle_weight=100,
                 ray_angle_threshold=60,
                 glass_weight=.01,
                 glass_catalog_path='./raytraced_optics/data/selected_ohara_glass.csv',
                 loss_multiplier=1,
                 **kwargs
                 ):
        super(RaytracedOptics, self).__init__(initial_lens_path, **kwargs)

        # Lens variable params
        self.quantized_continuous_glass_variables = quantized_continuous_glass_variables
        # Ray tracing params
        self.additional_rt_params = {}
        self.n_pupil_rings = n_pupil_rings
        self.n_ray_aiming_iter = n_ray_aiming_iter
        self.pupil_sampling = pupil_sampling
        self.wavelengths = {
            'R': wavelengths_r,
            'G': wavelengths_g,
            'B': wavelengths_b
        }
        assert len(set(len(w) for _, w in self.wavelengths.items())) == 1
        # Loss params
        self.ray_path_lower_thresholds = ray_path_lower_thresholds
        self.ray_path_upper_thresholds = ray_path_upper_thresholds
        self.ray_angle_threshold = ray_angle_threshold
        self.loss_weights = {
            'glass': glass_weight * loss_multiplier,
            'spot_size': spot_size_weight * loss_multiplier,
            'ray_path': ray_path_weight * loss_multiplier,
            'ray_angle': ray_angle_weight * loss_multiplier,
        }
        # Manage reference glasses
        ref_glasses = tf.constant(np.loadtxt(glass_catalog_path, delimiter=',', dtype=np.float32))
        self.catalog_g = tf.reshape(lm.g_from_n_v(*tf.unstack(ref_glasses, axis=1)), (-1, 2))

        self.initialize()

    def get_catalog_glass_indices(self):
        """
            Return the index of the closest catalog glass counterpart of each optimized glass
        """
        dist = tf.norm(self.g[:, None, :] / self.scale_factor - self.catalog_g[None, :, :], axis=-1)
        min_dist_idx = tf.argmin(dist, axis=1)
        return min_dist_idx

    def compute_losses(self, lens, rt_outputs):
        """
            From the outputs of the ray-tracing and the lens parameters,
                compute the loss that operates on the lens
        """
        x, y, *_, ray_ok, ray_backward, stacks = rt_outputs
        z_stack = tf.stack(stacks['z'], axis=0)
        ray_path_penalty = compute_ray_path_penalty(
            lens, z_stack, self.ray_path_lower_thresholds, self.ray_path_upper_thresholds)
        cos_squared = tf.stack(stacks['cos2'] + stacks['cos2_prime'])
        ray_angle_penalty = compute_ray_angle_penalty(cos_squared, self.ray_angle_threshold)
        self.loss_dict = {
            'glass': compute_glass_penalty(lens.structure, self.g / self.scale_factor, self.catalog_g),
            'spot_size': tf.reduce_mean(rt.compute_rms2d(x, y, ray_ok)),
            'ray_path': ray_path_penalty,
            'ray_angle': ray_angle_penalty,
        }
        self.logged_metrics.update({'loss/' + k: v for k, v in self.loss_dict.items()})

    def get_losses(self):
        weighted_losses = {k: self.loss_dict[k] * v for k, v in self.loss_weights.items() if v is not None}
        return weighted_losses

    def do_ray_tracing(self, lens=None, should_log=True):
        """
            Do the raw ray tracing, whose intermediate results are used to compute
                the spot size, spot diagrams (for PSFs), and penalty terms
        """
        specs = self.specs
        lens = lens or self.lens

        # Log some metrics on the lens
        if should_log:
            self.logged_metrics.update({
                'lens/defocus': lens.flat_t[-1] - lens.bfl[0],
                'lens/back_focal_length': lens.bfl[0],
                'lens/percentage_distortion': 100 * rt.compute_distortion(specs, lens, [1.])[0, 0],
                'lens/relative_illumination': rt.compute_relative_illumination(
                specs, lens, [0., 1.], None, n_ray_aiming_iter=1)[0, -1, 0]
            })

        fields = list(np.linspace(0, 1, self.n_fields))

        wavelengths_flat = [item for k in ('R', 'G', 'B') for item in self.wavelengths[k]]

        rt_params = dict(
            n_rays=(self.n_pupil_rings, 1), rel_fields=fields, vig_fn=None,
            n_ray_aiming_iter=self.n_ray_aiming_iter, wavelengths=wavelengths_flat, mode=self.pupil_sampling)
        rt_params.update(**self.additional_rt_params)
        ray_tracer = rt.RayTracer(**rt_params)
        rt_outputs = ray_tracer.trace_rays(specs, lens, aggregate=True)
        x, y, *_, ray_ok, ray_backward, stacks = rt_outputs

        self.compute_losses(lens, rt_outputs)

        # Log some ray tracing metrics
        if should_log:
            self.logged_metrics.update({
                'ray_tracing/ray_failures': tf.reduce_sum(tf.cast(~ray_ok, tf.float32)),
                'ray_tracing/backward_rays': tf.reduce_sum(tf.cast(ray_backward, tf.float32)),
                'ray_tracing/max_ray_aiming_error': tf.reduce_max(tf.abs(
                            rt.compute_ray_aiming_error(specs, lens, fields, None, 1, 'real'))),
            })

        return x, y

    def sample_optics_model(self):
        """
            Sample all required data for every field value via ray tracing
            This includes the psfs, distortion shifts, and relative illumination factors
            No particular image resolution or aspect ratio is considered at this step
        """
        specs = self.specs
        lens = self.lens
        x, y = self.do_ray_tracing(lens)

        # Compute the coordinates of the centers of the PSF grids
        if self.apply_distortion and not self.distortion_by_warping:
            # Center the grids on the paraxial 'y' intersections on the image plane
            y_center = rt.get_paraxial_heights_at_image_plane(specs, lens, np.linspace(0, 1, self.n_fields))
        else:
            # Center the grids on the spot centroid
            y_center = tf.reduce_mean(tf.reshape(y, (self.n_fields, -1)), axis=1)

        sampled_psfs, accounted_energy = sample_psfs(x, y, y_center, self.psf_shape, self.psf_increment)
        self.sampled_psfs = ensure_finite(sampled_psfs, 0.)
        self.logged_metrics['ray_tracing/lowest_accounted_energy'] = tf.reduce_min(accounted_energy)

        if self.distortion_by_warping and self.apply_distortion:
            self.sampled_distortion_shifts = ensure_finite(sample_distortion_shifts(specs, lens, y_center), 0.)

        if self.apply_relative_illumination:
            mean_wavelengths = [np.mean(v) for k, v in self.wavelengths.items()]
            self.sampled_relative_illumination = ensure_finite(sample_relative_illumination(
                specs, lens, self.n_fields, mean_wavelengths, None), 1.)


def ensure_finite(tensor, replace_val=0.):
    return tf.where(tf.math.is_finite(tensor), tensor, replace_val)


def linear_interpolation(soft_indices, values):
    soft_indices = tf.clip_by_value(soft_indices, 0, values.shape[0] - 1)
    upper = tf.cast(tf.math.ceil(soft_indices), tf.int32)
    lower = tf.cast(tf.math.floor(soft_indices), tf.int32)
    upper_values = tf.gather(values, upper)
    lower_values = tf.gather(values, lower)
    return lower_values * (1 - (soft_indices % 1)) + upper_values * (soft_indices % 1)


def get_psf_weights(grid_h, grid_w, field_map, n_fields):
    # Compute the PSF interpolation weights by using a weighted sum of the sampled psfs
    # For a PSF corresponding to a given patch,
    # the weights are proportional to the number of pixels that are closest to each field in that given patch
    img_h, img_w = field_map.shape
    ph, pw = int(np.round(img_h / grid_h)), int(np.round(img_w / grid_w))

    rows_0 = np.round(np.linspace(0, 1, grid_h) * (img_h - ph)).astype(int)
    cols_0 = np.round(np.linspace(0, 1, grid_w) * (img_w - pw)).astype(int)
    rows_1 = rows_0 + ph
    cols_1 = cols_0 + pw

    # Discretize the field map with integers
    discrete_field_map = tf.cast(tf.round(field_map * (n_fields - 1)), tf.int32)

    # Reshape the discrete field map into (n_psfs, rh, rw)
    # The image dimensions aren't necessarily a multiple of the psf grid dimensions, so we account for that
    patches = []
    for row_0, row_1 in zip(rows_0, rows_1):
        for col_0, col_1 in zip(cols_0, cols_1):
            patches.append(discrete_field_map[row_0:row_1, col_0:col_1])
    reshaped_field_map = tf.stack(patches)
    fields = tf.range(n_fields)
    weights = tf.reduce_mean(tf.cast(reshaped_field_map[..., None] == fields, tf.float32), axis=(1, 2))
    return weights


def compute_ray_path_penalty(lens, z_stack, min_thickness, max_thickness):
    """
        z_stack: z-coordinates of the rays across all surfaces [n_surface, n_lens, n_field, n_pupil, n_wavelength]
        min_thickness/max_thickness: tuple (float/None, float/None, float/None)
    """
    min_thickness = [value if value is not None else -np.inf for value in min_thickness]
    max_thickness = [value if value is not None else np.inf for value in max_thickness]
    min_t_air, min_t_glass, min_t_image = min_thickness
    max_t_air, max_t_glass, max_t_image = max_thickness
    ref_vertex_z = tf.cumsum(tf.concat((tf.reshape(lens.t, (-1,)), tf.zeros(1)), axis=0))
    abs_z = z_stack + tf.reshape(ref_vertex_z, (-1, 1, 1, 1, 1))
    delta_z = abs_z[1:] - abs_z[:-1]
    # Combine the thresholds for air and glass
    min_t_map = np.where(lens.structure.mask_G, min_t_glass, min_t_air)
    max_t_map = np.where(lens.structure.mask_G, max_t_glass, max_t_air)
    # Do the same for the surface before the image plane
    min_t_map[:, lens.structure.mask.sum(axis=1) - 1] = min_t_image
    max_t_map[:, lens.structure.mask.sum(axis=1) - 1] = max_t_image
    thickness_penalty = tf.maximum(min_t_map.reshape(-1, 1, 1, 1, 1) - delta_z, 0) + \
                        tf.maximum(delta_z - max_t_map.reshape(-1, 1, 1, 1, 1), 0)
    thickness_penalty = tf.reduce_sum(tf.reduce_mean(thickness_penalty, axis=(1, 2, 3, 4)))
    return thickness_penalty


def compute_ray_angle_penalty(cos_squared, angle_threshold):
    threshold = np.cos(np.deg2rad(angle_threshold)) ** 2
    return tf.reduce_sum(tf.reduce_mean(tf.maximum(threshold - cos_squared, 0), axis=(1, 2, 3, 4)))


def compute_glass_penalty(structure, g, catalog_g):
    if catalog_g is not None:
        dist = tf.norm(g[:, None, :] - catalog_g[None, :, :], axis=-1)
        min_dist = tf.reduce_min(dist, axis=1)
        agg = rt.mask_replace(structure.mask_G, tf.zeros_like(structure.mask_G, dtype=g.dtype), min_dist)
        glass_penalty = tf.reduce_sum(agg ** 2)
        return glass_penalty
    else:
        return tf.constant(0., dtype=tf.float32)


def sample_psfs(x, y, y_center, psf_size, psf_increment):
    """
        x, y: shape [1, n_fields, n_pupil, n_wavelengths]
        y_center: shape [n_fields], represents the y coordinates of the center of the sampled PSFs
    """
    # For each color channel, combine the rays from multiple wavelengths into the pupil dimension
    x = tf.transpose(x, (0, 1, 3, 2))  # [1, n_fields, n_channels, n_rays]
    y = tf.transpose(y, (0, 1, 3, 2))
    x = tf.reshape(x, (*x.shape[:2], 3, -1))
    y = tf.reshape(y, (*y.shape[:2], 3, -1))

    # Replicate every ray according to the 'x' symmetry
    x = tf.concat((x, -x), axis=3)
    y = tf.concat((y, y), axis=3)

    # Sample the PSFs
    *_, y_centroid, sampled_psfs, accounted_energy = \
        rt.compute_psf(x, y, n_bins=psf_size, increment=psf_increment, y_target=y_center)
    sampled_psfs = tf.transpose(sampled_psfs, (0, 2, 3, 1))
    sampled_psfs = tf.reverse(sampled_psfs, axis=(1,))  # Flip vertically

    return sampled_psfs, accounted_energy


def interpolate_psfs(sampled_psfs, field_map, psf_grid_shape):
    grid_h, grid_w = psf_grid_shape

    # Interpolate the sampled PSFs based on the coordinates of each patch
    psf_weights = get_psf_weights(grid_h, grid_w, field_map, sampled_psfs.shape[0])
    interpolated_psfs = tf.reduce_sum(psf_weights[..., None, None, None] * sampled_psfs, axis=1)
    return interpolated_psfs


def rotate_and_resize_psfs(interpolated_psfs, x_map, y_map, psf_grid_shape, resized_psf_shape):
    grid_h, grid_w = psf_grid_shape
    # Find the center of each patch in relative coordinates
    x_0, x_1 = x_map[0], x_map[-1]
    y_0, y_1 = y_map[0], y_map[-1]
    x_center = (np.arange(grid_w) + 0.5) / grid_w * (x_1 - x_0) + x_0
    y_center = (np.arange(grid_h) + 0.5) / grid_h * (y_1 - y_0) + y_0

    # Find the angle of the center of each patch
    angles = tf.reshape(tf.atan2(x_center[None, :], y_center[:, None]), (-1,))

    # Rotate the PSFs
    rotated = tfa.image.rotate(interpolated_psfs, -angles, interpolation='bilinear')

    # Resize the PSFs
    resized = tf.image.resize(rotated, resized_psf_shape, method='bilinear', antialias=True)
    psfs = resized / tf.reduce_sum(resized, axis=(1, 2), keepdims=True)

    return psfs[None, ...]


def sample_relative_illumination(specs, lens, n_fields, wavelengths, vig_fn=None):
    # Sample relative illumination factors at different field values
    fields = list(np.linspace(0, 1, n_fields))

    wavelength = [np.array(wavelengths).mean()]
    relative_illumination = rt.compute_relative_illumination(specs, lens, fields, vig_fn, 1, wavelength)[0, :, 0]
    return relative_illumination


def interpolate_relative_illumination(sampled_relative_illumination, field_map):
    # Linearly interpolate the sampled relative illumination factors and create a relative illumination map
    n_fields = sampled_relative_illumination.shape[0]
    relative_illumination_image = linear_interpolation(field_map * (n_fields - 1), sampled_relative_illumination)
    return relative_illumination_image


def sample_distortion_shifts(specs, lens, y_centroid):
    """
        y_centroid: y coordinates of the centroids for all fields (shape: [n_field])
        we assume every field value to be equidistant
    """
    # For each field, compute where the rays would hit if there were no distortion
    n_fields = y_centroid.shape[0]
    fields = np.linspace(0, 1, n_fields)
    y_ref = tf.squeeze(rt.get_paraxial_heights_at_image_plane(specs, lens, fields), 0)

    # Approximate the distortion shift
    sampled_distortion_shifts = (y_centroid - y_ref) / y_ref[-1]

    return sampled_distortion_shifts


def interpolate_distortion_shifts(sampled_distortion_shifts, x, y):
    # Interpolate the distortion shifts and compute the relative "x" and "y" shifts
    # "x" and "y" must be in relative object-space coordinates
    # E.g., x ** 2 + y ** 2 = 1 corresponds to the full field of view
    n_fields = sampled_distortion_shifts.shape[0]
    r = tf.sqrt(x ** 2 + y ** 2)
    angle_map = tf.atan2(y, x)
    distortion_shift_map = linear_interpolation(r * (n_fields - 1), sampled_distortion_shifts)

    # Compute the warped coordinates
    x_shift = distortion_shift_map * tf.cos(angle_map)
    y_shift = distortion_shift_map * tf.sin(angle_map)

    return x_shift, y_shift


def apply_distortion_by_warping(img, dist_x_coords, dist_y_coords):
    """
        img: [B, H, W, C]
        dist_x_coords, dist_y_coords: [H x W]
    """
    # Can be batched more efficiently by merging the batch size into the channel dimension
    b, h, w, c = img.shape
    img = tf.transpose(img, (1, 2, 0, 3))  # (H, W, B, C)

    img = tf.reshape(img, (1, h, w, -1))  # (1, H, W, BxC)
    distorted_image = image_ops.interpolate_bicubic(img, dist_x_coords, dist_y_coords, (h, w))
    img = tf.reshape(distorted_image, (h, w, b, c))  # (H, W, B, C)

    img = tf.transpose(img, (2, 0, 1, 3))  # (B, H, W, C)
    return img
