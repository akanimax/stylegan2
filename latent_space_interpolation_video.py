""" Demo script for running Random-latent space interpolation on the trained StyleGAN OR
    Show the effect of stochastic noise on a fixed image """

import argparse
import pickle
from math import sqrt

import cv2
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from dnnlib import tflib


def get_image(point, generator, truncation_psi=0.7, resize=None, randomize_noise=False):
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    point = np.expand_dims(point, axis=0)
    gen_image = generator.run(
        point,
        None,
        truncation_psi=truncation_psi,
        randomize_noise=randomize_noise,
        output_transform=fmt,
    )

    img = np.squeeze(gen_image, axis=0)

    if resize is not None:
        img = imresize(img, resize, interp="bicubic")

    return img


def parse_arguments():
    parser = argparse.ArgumentParser(
        "StyleGANv2 image_generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "pickle_file",
        type=str,
        action="store",
        help="pickle file containing the trained styleGAN model",
    )

    parser.add_argument(
        "output_file",
        type=str,
        default="latent_space_exploration.mpeg",
        action="store",
        help="output video file",
    )

    parser.add_argument(
        "--random_state",
        action="store",
        type=int,
        default=5,
        help="random_state (seed) for the script to run",
    )

    parser.add_argument(
        "--num_points",
        action="store",
        type=int,
        default=21,
        help="Number of samples to be seen",
    )

    parser.add_argument(
        "--transition_points",
        action="store",
        type=int,
        default=60,
        help="Number of transition samples for interpolation. Can also be considered as fps",
    )

    parser.add_argument(
        "--resize",
        action="store",
        default=None,
        nargs=2,
        required=False,
        help="Resolutions used for generating the interpolation",
    )

    parser.add_argument(
        "--smoothing",
        action="store",
        type=float,
        default=1.0,
        help="amount of transitional smoothing",
    )

    parser.add_argument(
        "--only_noise",
        action="store",
        type=bool,
        default=False,
        help="to visualize the same point with only different realizations of noise",
    )

    parser.add_argument(
        "--truncation_psi",
        action="store",
        type=float,
        default=0.6,
        help="value of truncation_psi used for generating the video",
    )

    parser.add_argument(
        "--fps", action="store", type=int, default=24, help="fps of the generated video"
    )

    args = parser.parse_args()
    return args


def main(args):
    # Initialize TensorFlow.
    tflib.init_tf()

    with open(args.pickle_file, "rb") as f:
        _, _, Gs = pickle.load(f)

    # Print network details.
    print("\n\nLoaded the Generator as:")
    Gs.print_layers()

    # Pick latent vector.
    latent_size = Gs.input_shape[1]
    rnd = np.random.RandomState(args.random_state)

    # create the random latent_points for the interpolation
    total_frames = args.num_points * args.transition_points
    all_latents = rnd.randn(total_frames, latent_size)
    all_latents = gaussian_filter(
        all_latents, [args.smoothing * args.transition_points, 0], mode="wrap"
    )
    all_latents = (
        all_latents / np.linalg.norm(all_latents, axis=-1, keepdims=True)
    ) * sqrt(latent_size)

    # handling the latent points
    start_point = all_latents[0]
    points = all_latents[:]

    # if we have only noise realization, then all points are start_point
    if args.only_noise:
        points = np.array([start_point for _ in points])

    # handle the dynamic inputs
    resize = args.resize
    if resize is not None:
        resize = [int(val) for val in resize]

    # make the video:
    sample_image_for_shape = get_image(
        start_point,
        Gs,
        truncation_psi=args.truncation_psi,
        resize=resize,
        randomize_noise=args.only_noise,
    )
    height, width, _ = sample_image_for_shape.shape

    video = cv2.VideoWriter(
        args.output_file, cv2.VideoWriter_fourcc(*"MP4V"), args.fps, (width, height)
    )

    for point in tqdm(points):
        image = get_image(
            point,
            Gs,
            truncation_psi=args.truncation_psi,
            resize=resize,
            randomize_noise=args.only_noise,
        )
        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()

    print(f"Video created at: {args.output_file}")


if __name__ == "__main__":
    main(parse_arguments())
