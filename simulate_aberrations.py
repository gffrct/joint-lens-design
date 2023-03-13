"""
    Sample script to simulate realistic aberrations on a given image
"""
import tensorflow as tf
import imageio
import os
import argparse

import optics_simulator


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Aberration simulator for spherical compound lenses based on exact ray tracing')
    parser.add_argument('--input-image', default='./data/sample_image.png',
                        help='Path of the input image')
    parser.add_argument('--output-image-path', default='./data/output_image.png',
                        help='Path of the output image')
    parser.add_argument('--lens-path', default='./data/baseline_doublet.yml',
                        help='Path of the lens file (.yml)')
    parser.add_argument('--glass-catalog-path', default='./data/selected_ohara_glass.csv',
                        help='Path of the glass catalog file (.csv)')
    parser.add_argument('--simulated-resolution-factor', default=1,
                        help='Simulated resolution factor (recommended values: 1 or 2)')
    args = parser.parse_args()

    model = optics_simulator.RaytracedOptics(
        initial_lens_path=args.lens_path,
        glass_catalog_path=args.glass_catalog_path,
        n_pupil_rings=24,  # Down from 32 (default) to fit into memory
        n_sampled_fields=11,  # Down from 21 (default) to fit into memory
        simulated_res_factor=args.simulated_resolution_factor,
        apply_distortion=True,
        apply_relative_illumination=True,
        lazy_init=True
    )

    img = imageio.imread(args.input_image)

    img = tf.constant(tf.cast(img[None, ...], tf.float32))

    out_img = model(img)
    out_img = tf.cast(tf.clip_by_value(out_img[0], 0., 255.), tf.uint8)

    output_path = args.output_image_path
    imageio.imsave(output_path, out_img)
    link = os.path.abspath(output_path).replace('\\', '/')
    print(f'\nLocation of the output image: file:///{link}')
