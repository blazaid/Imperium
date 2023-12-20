import math
import random

from enum import Enum

from PIL import Image
from PIL import ImageDraw

from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class SpectralType(Enum):
    O = (0x9B / 0xFF, 0xB0 / 0xFF, 0xFF / 0xFF, 1)
    B = (0xAA / 0xFF, 0xBF / 0xFF, 0xFF / 0xFF, 1)
    A = (0xCA / 0xFF, 0xD7 / 0xFF, 0xFF / 0xFF, 1)
    F = (0xF8 / 0xFF, 0xF7 / 0xFF, 0xFF / 0xFF, 1)
    G = (0xFF / 0xFF, 0xF4 / 0xFF, 0xEA / 0xFF, 1)
    K = (0xFF / 0xFF, 0xD2 / 0xFF, 0xA1 / 0xFF, 1)
    M = (0xFF / 0xFF, 0xCC / 0xFF, 0x6F / 0xFF, 1)

    def __init__(self, r, g, b, a):
        self.color = r, g, b, a


class Star:
    def __init__(self, x: float, y: float, z: float, st: SpectralType):
        self.x = x
        self.y = y
        self.z = z
        self.st = st


def plot(stars):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 1))

    xs = [star.x for star in stars]
    ys = [star.y for star in stars]
    zs = [star.z for star in stars]
    colors = [star.st.color for star in stars]

    ax.scatter(xs, ys, zs, s=0.5, c=colors)


def cartesian():
    pass


def random_star(x, y, z):
    return Star(
        x=x,
        y=y,
        z=z,
        st=random.choices(
            list(SpectralType),
            weights=(W_O, W_B, W_A, W_F, W_G, W_K, W_M)
        )[0]
    )

# Star type rates
W_O = 1
W_B = 1
W_A = 1
W_F = 1
W_G = 1
W_K = 1
W_M = 1

NAME = 'Galaxy'
NUMHUB = 2000  # Number of Core Stars
NUMDISK = 4000  # Number of Disk Stars
HUBRAD = 45.0  # Radius of Core
DISKRAD = 45.0  # Radius of Disk
ARMROTS = 0.5  # Tightness of Arm Winding
ARMWIDTH = 65  # Arm Width in Degrees
MAXHUBZ = 16  # Maximum Depth of Core
MAXDISKZ = 1.0  # Maximum Depth of Arms
FUZZ = 25  # Maximum Outlier Distance from Arms

SHRAD = HUBRAD * 0.1
SDRAD = DISKRAD * 0.1


def spiral_galaxy(*, num_arms):
    stars = []

    '''
    # omega is the separation (in degrees) between each arm
    # Prevent div by zero error:
    omega = 360.0 / num_arms if num_arms else 0.0
    for _ in range(NUMDISK):
        # Choose a random distance from center
        dist = HUBRAD + random.random() * DISKRAD
        distb = dist + random.uniform(0, SDRAD)

        # This is the 'clever' bit, that puts a star at a given distance
        # into an arm: First, it wraps the star round by the number of
        # rotations specified.  By multiplying the distance by the number of
        # rotations the rotation is proportional to the distance from the
        # center, to give curvature
        theta = ((360.0 * ARMROTS * (distb / DISKRAD))

                 # Then move the point further around by a random factor up to
                 # ARMWIDTH
                 + random.random() * ARMWIDTH

                 # Then multiply the angle by a factor of omega, putting the
                 # point into one of the arms
                 + omega * random.randrange(0, num_arms)

                 # Then add a further random factor, 'fuzzin' the edge of the arms
                 + random.random() * FUZZ * 2.0 - FUZZ
                 # + random.randrange( -FUZZ, FUZZ )
                 )

        # Convert to cartesian
        # def cartesian_convert
        x = math.cos(theta * math.pi / 180.0) * distb
        y = math.sin(theta * math.pi / 180.0) * distb
        z = random.random() * MAXDISKZ * 2.0 - MAXDISKZ

        # Add star to the stars array
        stars.append(random_star(x=x, y=y, z=z))
    '''
    # Now generate the Hub. This places a point on or under the curve
    # maxHubZ - s d^2 where s is a scale factor calculated so that z = 0 is
    # at maxHubR (s = maxHubZ / maxHubR^2) AND so that minimum hub Z is at
    # maximum disk Z. (Avoids edge of hub being below edge of disk)

    # First, we generate the galaxy hub as a
    scale = MAXHUBZ / (HUBRAD * HUBRAD)
    i = 0
    while i < NUMHUB:
        # Choose a random distance from center
        dist = random.uniform(0, 1) * HUBRAD
        distb = dist + random.uniform(0, SHRAD)

        # Any rotation (points are not on arms)
        theta = random.random() * 360

        # Convert to cartesian
        x = math.cos(theta * math.pi / 180.0) * distb
        y = math.sin(theta * math.pi / 180.0) * distb
        z = (random.random() * 2 - 1) * (MAXHUBZ - scale * distb * distb)

        stars.append(random_star(x=x, y=y, z=z))

        # Process next star
        i = i + 1

    return stars

# Generate the galaxy
stars = spiral_galaxy(
    num_arms=2
)

# Save the galaxy as PNG to galaxy.png
plot(stars)