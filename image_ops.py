import tensorflow as tf
import numpy as np


def svola_convolution(image, overlap_size, psfs, psfs_grid_shape, window_type='boxcar'):
    """
        image [B, H, W, C], psfs [B, N, H, W, C] (N is the number of psfs depending on the psf grid size)
        window_type is the type of 2D separable window function used; either "boxcar" (default) or "hann"
    """
    if isinstance(overlap_size, int):
        overlap_size = (overlap_size, overlap_size)
    n_img, im_h_orig, im_w_orig, n_channels = image.shape
    n_patches, kh, kw = psfs.shape[1:4]
    im_h = im_h_orig + 2 * overlap_size[0]
    im_w = im_w_orig + 2 * overlap_size[1]
    assert kh % 2 == 1 and kw % 2 == 1
    pad_h = kh // 2
    pad_w = kw // 2
    total_pad_h = overlap_size[0] + pad_h
    total_pad_w = overlap_size[1] + pad_w

    paddings = ((0, 0), (total_pad_h, total_pad_h), (total_pad_w, total_pad_w), (0, 0))
    image = tf.pad(image, paddings, "SYMMETRIC")

    patch_size = (im_h_orig // psfs_grid_shape[0] + overlap_size[0] * 2,
                  im_w_orig // psfs_grid_shape[1] + overlap_size[1] * 2)

    # Compute the beginning and end coordinates of all image patches
    # Padding due to overlap is considered, but not the one due to the kernel size
    # If the image shape is not a multiple of the grid shape, we stretch those coordinates
    # so that outside patches start or end at the (padded) border
    rows_0 = np.round(np.linspace(0, 1, psfs_grid_shape[0]) * (im_h - patch_size[0])).astype(int)
    cols_0 = np.round(np.linspace(0, 1, psfs_grid_shape[1]) * (im_w - patch_size[1])).astype(int)
    rows_1 = rows_0 + patch_size[0]
    cols_1 = cols_0 + patch_size[1]
    rows_0, cols_0 = np.meshgrid(rows_0, cols_0, indexing='ij')
    rows_1, cols_1 = np.meshgrid(rows_1, cols_1, indexing='ij')
    patch_corners = list(zip(rows_0.ravel(), rows_1.ravel(), cols_0.ravel(), cols_1.ravel()))

    patches = tf.stack([image[:, r0:r1 + 2 * pad_h, c0:c1 + 2 * pad_w, :] for r0, r1, c0, c1 in patch_corners], axis=0)

    # Transpose the patches from [N, B, H, W, C]
    patches = tf.transpose(patches, perm=[0, 1, 4, 2, 3])  # [N, B, C, H, W]
    ph, pw = patches.shape[-2:]

    # Pad the PSFs and transpose
    psf_paddings = ((0, 0), (0, 0), (0, ph - kh), (0, pw - kw), (0, 0))
    psfs = tf.pad(psfs, psf_paddings, 'CONSTANT')
    psfs = tf.transpose(psfs, perm=[1, 0, 4, 2, 3])  # [N, B, C, H, W]

    # Do convolution in Fourier space
    patches = tf.cast(patches, tf.complex64)
    patches = tf.signal.fft2d(patches)
    psfs = tf.cast(psfs, tf.complex64)
    psfs = tf.signal.fft2d(psfs)
    patches = tf.multiply(patches, psfs)
    patches = tf.signal.ifft2d(patches)
    patches = tf.abs(patches)
    patches = tf.roll(patches, shift=[-(pad_h + 1), -(pad_w + 1)], axis=[3, 4])

    # Transpose and crop the paddings (need to reshape first)
    patches = tf.transpose(patches, perm=[0, 1, 3, 4, 2])  # [N, B, H, W, C]
    patches = tf.reshape(patches, (-1, *patches.shape[-3:]))  # [N x B, H, W, C]
    patches = tf.image.resize_with_crop_or_pad(patches, *patch_size)
    patches = tf.reshape(patches, (n_patches, -1, *patches.shape[-3:]))  # [N, B, H, W, C]

    # Compute the normalized weights (contribution for each pixel of a patch to the final image)
    window_fn = {
        'boxcar': lambda x: np.ones(x.shape),
        'hann': lambda x: np.sin(np.pi * x) ** 2
    }
    row_window = window_fn[window_type](np.linspace(0, 1, patch_size[0] + 2)[1:-1])
    col_window = window_fn[window_type](np.linspace(0, 1, patch_size[1] + 2)[1:-1])
    window = row_window[:, None] * col_window[None, :]
    im_patch_weights = []
    for r0, r1, c0, c1 in patch_corners:
        im_patch_w = np.zeros((im_h, im_w, 1)).astype(np.float32)
        im_patch_w[r0:r1, c0:c1, 0] = window
        im_patch_weights.append(im_patch_w)
    normalized_weights_padded = im_patch_weights / (np.sum(np.array(im_patch_weights), axis=0))

    im_out = tf.zeros((n_img, im_h, im_w, n_channels))
    for patch, weights, (r0, r1, c0, c1) in zip(tf.unstack(patches), normalized_weights_padded, patch_corners):
        patch_weights = weights[r0:r1, c0:c1]
        weighted_patch = tf.multiply(patch, tf.constant(patch_weights))

        vertical_padding = [r0, im_h - r1]
        horizon_padding = [c0, im_w - c1]
        paddings = ((0, 0), vertical_padding, horizon_padding, (0, 0))

        padded_weighted_patch = tf.pad(weighted_patch, paddings, 'CONSTANT')

        # Accumulate the results from every patch to limit memory use
        im_out = im_out + padded_weighted_patch

    im_out = tf.image.crop_to_bounding_box(im_out, overlap_size[0], overlap_size[1], im_h_orig, im_w_orig)
    return im_out


def repeat(x, n_repeats):
    rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
    return tf.reshape(rep, [-1])


def interpolate_bicubic(im, x, y, out_size):
    # Adapted from https://github.com/dantkz/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    # New implementation is more memory efficient

    alpha = -0.75  # same as in tf.image.resize_images, see tensorflow/tensorflow/core/kernels/resize_bicubic_op.cc
    bicubic_coeffs = (
        (1, 0, -(alpha + 3), (alpha + 2)),
        (0, alpha, -2 * alpha, alpha),
        (0, -alpha, 2 * alpha + 3, -alpha - 2),
        (0, 0, alpha, -alpha)
    )

    batch_size, height, width, channels = im.get_shape().as_list()

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    out_height = out_size[0]
    out_width = out_size[1]

    # Scale indices from [-1, 1] to [0, width/height - 1]
    x = tf.clip_by_value(x, -1, 1)
    y = tf.clip_by_value(y, -1, 1)
    x = (x + 1.0) / 2.0 * (width_f - 1.0)
    y = (y + 1.0) / 2.0 * (height_f - 1.0)

    # Do sampling
    # Integer coordinates of 4x4 neighbourhood around (x0_f, y0_f)
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)
    xm1_f = x0_f - 1
    ym1_f = y0_f - 1
    xp1_f = x0_f + 1
    yp1_f = y0_f + 1
    xp2_f = x0_f + 2
    yp2_f = y0_f + 2

    # Clipped integer coordinates
    xs = [0, 0, 0, 0]
    ys = [0, 0, 0, 0]
    xs[0] = tf.cast(x0_f, tf.int32)
    ys[0] = tf.cast(y0_f, tf.int32)
    xs[1] = tf.cast(tf.maximum(xm1_f, 0), tf.int32)
    ys[1] = tf.cast(tf.maximum(ym1_f, 0), tf.int32)
    xs[2] = tf.cast(tf.minimum(xp1_f, width_f - 1), tf.int32)
    ys[2] = tf.cast(tf.minimum(yp1_f, height_f - 1), tf.int32)
    xs[3] = tf.cast(tf.minimum(xp2_f, width_f - 1), tf.int32)
    ys[3] = tf.cast(tf.minimum(yp2_f, height_f - 1), tf.int32)

    # Indices of neighbours for the batch
    dim2 = width
    dim1 = width * height
    base = repeat(tf.range(batch_size) * dim1, out_height * out_width)

    idx = []
    for i in range(4):
        idx.append([])
        for j in range(4):
            cur_idx = base + ys[i] * dim2 + xs[j]
            idx[i].append(cur_idx)

    # Use indices to lookup pixels in the flat image and restore channels dim
    im_flat = tf.reshape(im, [-1, channels])

    def get_weights(x, x0_f):
        tx = (x - x0_f)
        tx2 = tx * tx
        tx3 = tx2 * tx
        t = [1, tx, tx2, tx3]
        weights = []
        for i in range(4):
            result = 0
            for j in range(4):
                result = result + bicubic_coeffs[i][j] * t[j]
            result = tf.reshape(result, [-1, 1])
            weights.append(result)
        return weights

    x_weights = get_weights(x, x0_f)
    y_weights = get_weights(y, y0_f)
    output = tf.zeros_like(im_flat)
    for i in range(4):
        x_interp = tf.zeros_like(im_flat)
        for j in range(4):
            # To calculate interpolated values first, interpolate in x dim 4 times for y=[0, -1, 1, 2]
            x_interp = x_interp + x_weights[j] * tf.gather(im_flat, idx[i][j])
        # Finally, interpolate in y dim using interpolations in x dim
        output = output + y_weights[i] * x_interp

    return output
