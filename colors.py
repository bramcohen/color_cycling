# This file is public domain

import numpy as np

def distance_preserving_matrix(A):
    U, S, V = np.linalg.svd(A)
    return U @ V

def test_distance_preserving_matrix():
    # Test a 3x3 matrix
    A = np.array(np.random.normal(size=(3, 3)))
    M = distance_preserving_matrix(np.random.normal(size=(3, 3)))
    # Test that the distance between the first two rows of A
    # is the same as the distance between the first two rows of M*A
    assert np.allclose(np.linalg.norm(A[0,:] - A[1,:]), np.linalg.norm(M @ A[0,:] - M @ A[1,:]))
    # Test that the distance between the second and third rows of A
    # is the same as the distance between the second and third rows of M*A
    assert np.allclose(np.linalg.norm(A[1,:] - A[2,:]), np.linalg.norm(M @ A[1,:] - M @ A[2,:]))

def test_distance_preserving_matrix2():
    Q1 = distance_preserving_matrix(np.random.normal(size=(3, 3)))
    Q2 = distance_preserving_matrix(Q1)
    assert np.linalg.norm(Q2 - Q1) < 1e-12

test_distance_preserving_matrix()
test_distance_preserving_matrix2()

def make_rotation_matrix(axis, angle_degrees):
    angle = np.radians(angle_degrees)
    axis = axis / np.linalg.norm(axis)
    axis_cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * np.cos(angle) + np.sin(angle) * axis_cross + (1 - np.cos(angle)) * np.outer(axis, axis)

def rotate_vector_3d(v, axis, angle_degrees):
    return make_rotation_matrix(axis, angle_degrees) @ v

def test_rotate_vector():
    for i in range(1, 5):
        v = np.random.normal(size=[3])
        R = make_rotation_matrix(np.random.normal(size=[3]), 360 / i)
        v2 = v
        for j in range(i):
            v2 = R @ v2
        assert np.linalg.norm(v2 - v) < 1e-12

test_rotate_vector()

def gamma(b):
    if b <= 0.04045:
        return b / 12.92
    return ((b + 0.055)/1.055) ** 2.4

def ungamma(b):
    if b <= 0.0031308:
        return b * 12.92
    return (b ** (1/2.4)) * 1.055 - 0.055

def test_gamma():
    for i in range(51):
        x = i / 50
        assert -1e-12 < ungamma(gamma(x)) - x < 1e-12

test_gamma()

def cube_to_sphere(v):
    assert np.min(v) >= 0
    assert np.max(v) <= 1
    v2 = np.array([gamma(x) for x in v])
    v2 = (v2 - 0.5) * 2
    v2 /= np.linalg.norm(v2/np.max(np.abs(v2)))
    return v2

def sphere_to_cube(v):
    v2 = v * np.linalg.norm(v/np.max(np.abs(v)))
    v2 = (v2/2) + 0.5
    return np.array([ungamma(x) for x in v2])

def test_cube_sphere():
    for i in range(20):
        v = np.random.rand(3)
        assert np.linalg.norm(v - sphere_to_cube(cube_to_sphere(v))) < 1e-12

test_cube_sphere()

from PIL import Image
import imageio.v3 as imageio

def dither(start_image):
    new_image = np.zeros(start_image.shape, dtype='uint8')
    maxx = start_image.shape[0] - 1
    maxy = start_image.shape[1] - 1
    for x in range(start_image.shape[0]):
        for y in range(start_image.shape[1]):
            for z in range(3):
                old_val = start_image[x][y][z]
                closest = max(0, min(int(old_val + 0.5), 255))
                new_image[x][y][z] = closest
                error = old_val - closest
                if x < maxx:
                    start_image[x+1][y][z] += error * 7 / 16
                if x > 0 and y < maxy:
                    start_image[x-1][y+1][z] += error * 3 / 16
                if y < maxy:
                    start_image[x][y+1][z] += error * 5 / 16
                if x < maxx and y < maxy:
                    start_image[x+1][y+1][z] += error / 16
    return new_image

# This cycles around a specified vector assuming that the origin is gray.
# Automatically detects file output type from name extension. 'loop' is 
# used with .gif
# This tends to produce negative image effects if the vector is too far off 
# from [1, 1, 1]
# It suffers from cognitive distance in luminance being much greater than
# across hues
def cycle_image(input_filename, output_filename, vector, steps, fps, loop = None):
    rotation_matrix = make_rotation_matrix(vector, 360 / steps)
    my_image = np.array(Image.open(input_filename).convert('RGB', dither=None, 
            palette=Image.ADAPTIVE, colors=256), dtype='uint8')
    images = [my_image]
    last_image = np.zeros(my_image.shape, dtype=float)
    for x in range(last_image.shape[0]):
        for y in range(last_image.shape[1]):
            last_image[x, y, :] = cube_to_sphere(my_image[x, y, :] / 256)
    for i in range(steps - 1):
        for x in range(last_image.shape[0]):
            for y in range(last_image.shape[1]):
                last_image[x, y, :] = rotation_matrix @ last_image[x, y, :]
                assert np.linalg.norm(last_image[x, y, :]) <= 1 + 1e-12
        new_image = np.zeros(my_image.shape, dtype=float)
        for x in range(last_image.shape[0]):
            for y in range(last_image.shape[1]):
                new_image[x, y, :] = sphere_to_cube(last_image[x, y, :]) * 256
        images.append(dither(new_image))
    if loop is not None:
        imageio.imwrite(output_filename, images, fps=fps, loop=loop)
    else:
        imageio.imwrite(output_filename, images, fps=fps)

from coloraide import Color
from coloraide.spaces.okhsl import Okhsl
Color.register(Okhsl())

def flip_rotate(coords, stepnum, steps):
    angle = 360 * stepnum / steps
    h, s, l = Color("srgb", [x/255 for x in coords]).convert("okhsl").coords()
    r = [x * 256 for x in Color('okhsl', [(angle-h) % 360, s, l]).convert('srgb').coords()]
    return r

# Produces a color mirroring effect. At any given point two opposite hues will 
# be true and the two and 90 degrees from those will be flipped.
# Automatically detects file output type from name extension. 'loop' is 
# used with .gif
def cycle_image2(input_filename, output_filename, steps, fps, loop = None):
    my_image = np.array(Image.open(input_filename).convert('RGB', dither=None, 
            palette=Image.ADAPTIVE, colors=256), dtype='uint8')
    images = []
    for i in range(steps):
        new_image = np.zeros(my_image.shape, dtype=float)
        for x in range(my_image.shape[0]):
            for y in range(my_image.shape[1]):
                new_image[x, y, :] = flip_rotate(my_image[x, y, :], i, steps)
        images.append(dither(new_image))
    if loop is not None:
        imageio.imwrite(output_filename, images, fps=fps, loop=loop)
    else:
        # Yes this library will puke if you set loop=None when it's unexpected
        imageio.imwrite(output_filename, images, fps=fps)
