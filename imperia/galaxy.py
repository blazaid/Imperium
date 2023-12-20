import abc
import enum
import math
import random
from typing import List

from astropy import units as u

__all__ = [
    'Moon',
    'Planet',
    'Star',
    'System',
    'Galaxy',
    'GalaxyBuilder',
]

from astropy.coordinates import SkyCoord


class AstronomicalObject(metaclass=abc.ABCMeta):
    """General behaviour for any astronomical object in the galaxy."""


class Moon(AstronomicalObject):
    """A moon belonging to a planet."""

    def __str__(self):
        return 'moon'


class Planet(AstronomicalObject):
    def __init__(self, *, moons: List[Moon] = None):
        """Initializes this object.

        :param moons: The list of moons that belong to this planet.
        """
        self.moons = moons or []

    def __str__(self):
        if self.moons:
            moons = ', '.join(str(moon) for moon in self.moons)
            return f'planet (moons: {moons})'
        else:
            return 'planet'


class SpectralType(enum.Enum):
    """Spectral type according to the Harvard classification.

    Temperature is measured in Kelvin degrees, mass in solar masses and
    radius in solar radii.
    """
    O = (30000, 100000, 16, 300, 6.6, 1700, 0x9BFFB0)
    B = (10000, 30000, 2.1, 16, 1.8, 6.6, 0xAABFFF)
    A = (7500, 10000, 1.4, 2.1, 1.4, 1.8, 0xCAD7FF)
    F = (6000, 7500, 1.04, 1.4, 1.15, 1.4, 0xF8F7FF)
    G = (5200, 6000, 0.8, 1.04, 0.96, 1.15, 0xFFF4EA)
    K = (3700, 5200, 0.45, 0.8, 0.7, 0.96, 0xFFD2A1)
    M = (2400, 3700, 0.08, 0.45, 0, 0.7, 0xFFCC6F)

    def __init__(self, min_temp, max_temp, min_mass, max_mass, min_radius, max_radius, chromaticity):
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.chromaticity = chromaticity


class Star(AstronomicalObject):
    """A star belonging to an stellar system."""

    def __init__(self, *, name, spectral_type, temperature, mass, radius):
        self.__name = name
        self.__spectral_type = spectral_type
        self.__temp = temperature
        self.__mass = mass
        self.__radius = radius

    @property
    def name(self):
        return self.__name

    @property
    def spectral_type(self):
        return self.__spectral_type

    @property
    def temp(self):
        return self.__temp * u.Kelvin

    @property
    def mass(self):
        return self.__mass * u.solMass

    @property
    def radius(self):
        return self.__radius * u.solRad

    def __str__(self):
        mass = f'{self.mass.value:.3f} M☉'
        radius = f'{self.radius.value:.3f} R☉'
        temp = f'{self.temp.value:.3f} K'
        return f'{self.name} ({mass}, {radius}, {temp})'


class StarBuilder:
    def __init__(self, *, config):
        # Proportion for the different kind of stars in the galaxy
        self.__type_O_proportion = None
        self.type_O_proportion = config.getfloat(
            'Stars', 'Spectral type O proportion'
        )
        self.__type_B_proportion = None
        self.type_B_proportion = config.getfloat(
            'Stars', 'Spectral type B proportion'
        )
        self.__type_A_proportion = None
        self.type_A_proportion = config.getfloat(
            'Stars', 'Spectral type A proportion'
        )
        self.__type_F_proportion = None
        self.type_F_proportion = config.getfloat(
            'Stars', 'Spectral type F proportion'
        )
        self.__type_G_proportion = None
        self.type_G_proportion = config.getfloat(
            'Stars', 'Spectral type G proportion'
        )
        self.__type_K_proportion = None
        self.type_K_proportion = config.getfloat(
            'Stars', 'Spectral type K proportion'
        )
        self.__type_M_proportion = None
        self.type_M_proportion = config.getfloat(
            'Stars', 'Spectral type M proportion'
        )
        # Total (to compute the proportions)
        self.__total_proportions = None

    ####################################################################
    # Configuration
    #
    @property
    def type_O_proportion(self):
        if self.__type_O_proportion is not None:
            return self.__type_O_proportion / self.total_proportions
        else:
            return 0

    @type_O_proportion.setter
    def type_O_proportion(self, value):
        self.__type_O_proportion = value

    @property
    def type_B_proportion(self):
        if self.__type_B_proportion is not None:
            return self.__type_B_proportion / self.total_proportions
        else:
            return 0

    @type_B_proportion.setter
    def type_B_proportion(self, value):
        self.__type_B_proportion = value

    @property
    def type_A_proportion(self):
        if self.__type_A_proportion is not None:
            return self.__type_A_proportion / self.total_proportions
        else:
            return 0

    @type_A_proportion.setter
    def type_A_proportion(self, value):
        self.__type_A_proportion = value

    @property
    def type_F_proportion(self):
        if self.__type_F_proportion is not None:
            return self.__type_F_proportion / self.total_proportions
        else:
            return 0

    @type_F_proportion.setter
    def type_F_proportion(self, value):
        self.__type_F_proportion = value

    @property
    def type_G_proportion(self):
        if self.__type_G_proportion is not None:
            return self.__type_G_proportion / self.total_proportions
        else:
            return 0

    @type_G_proportion.setter
    def type_G_proportion(self, value):
        self.__type_G_proportion = value

    @property
    def type_K_proportion(self):
        if self.__type_K_proportion is not None:
            return self.__type_K_proportion / self.total_proportions
        else:
            return 0

    @type_K_proportion.setter
    def type_K_proportion(self, value):
        self.__type_K_proportion = value

    @property
    def type_M_proportion(self):
        if self.__type_M_proportion is not None:
            return self.__type_M_proportion / self.total_proportions
        else:
            return 0

    @type_M_proportion.setter
    def type_M_proportion(self, value):
        self.__type_M_proportion = value

    @property
    def total_proportions(self):
        if self.__total_proportions is None:
            self.__total_proportions = sum(
                value for value in (
                    self.__type_O_proportion or 0,
                    self.__type_B_proportion or 0,
                    self.__type_A_proportion or 0,
                    self.__type_F_proportion or 0,
                    self.__type_G_proportion or 0,
                    self.__type_K_proportion or 0,
                    self.__type_M_proportion or 0,
                )
            )
        return self.__total_proportions

    ####################################################################
    # Helper properties
    #
    @property
    def star_types_and_proportions(self):
        return (
            (SpectralType.O, self.type_O_proportion),
            (SpectralType.B, self.type_B_proportion),
            (SpectralType.A, self.type_A_proportion),
            (SpectralType.F, self.type_F_proportion),
            (SpectralType.G, self.type_G_proportion),
            (SpectralType.K, self.type_K_proportion),
            (SpectralType.M, self.type_M_proportion),
        )

    ####################################################################
    # Builders
    #
    def build(self, star_name=None, spectral_types=None):
        """Creates a random star.

        :return: A newly created star.
        """
        # Name. It'll be a newly generated name or an existent name if
        # specified
        # TODO Star name generator here
        name = star_name or 'star'
        # Spectral type, selected based on their proportion. If the list
        # of spectral types is filtered, then only those specified have
        # the chance to be selected
        types, weights = zip(*(
            (spectral_type, proportion)
            for (spectral_type, proportion) in self.star_types_and_proportions
            if spectral_types is None or spectral_type in spectral_types
        ))
        spectral_type = random.choices(types, weights=weights)[0]
        # Temperature
        sigma = (spectral_type.max_temp - spectral_type.min_temp) / 2
        mu = spectral_type.min_temp + sigma
        temperature = random.gauss(mu, sigma)
        # Mass
        sigma = (spectral_type.max_mass - spectral_type.min_mass) / 2
        mu = spectral_type.min_mass + sigma
        mass = random.gauss(mu, sigma)
        # Radius
        sigma = (spectral_type.max_radius - spectral_type.min_radius) / 2
        mu = spectral_type.min_radius + sigma
        radius = random.gauss(mu, sigma)
        return Star(
            name=name,
            spectral_type=spectral_type,
            temperature=temperature,
            mass=mass,
            radius=radius,
        )


class System(AstronomicalObject):
    """A system in the galaxy.

    It can be either a stellar system or a planetary system, depending
    of the existence of planets in it.
    """

    def __init__(self, *, position, stars, planets):
        """Initializes this object.

        :param position: The coordinates of this system in the galaxy.
        :param stars: The stars in this system.
        :param planets: The planets in this system.
        """
        self.position = position
        self.stars = stars
        self.planets = planets

    def color(self):
        max(self.stars, key=lambda star: star.mass).color()

    def __str__(self):
        stars = ', '.join(str(star) for star in self.stars)
        if self.planets:
            planets = ', '.join(str(planet) for planet in self.planets)
        else:
            planets = 'No planets'
        return f'System\n\t· Stars: {stars}\n\t· Planets: {planets}'


class Galaxy:
    """The place where the action of the game take place."""

    def __init__(self, *, systems):
        self.systems = systems

    def __str__(self):
        return '\n'.join(str(system) for system in self.systems)


class GalaxyBuilder:
    def __init__(self, *, config):
        self.star_builder = StarBuilder(config=config)

        # Minimum distance allowed between two systems
        self.__min_dist_between_systems = None
        self.minimum_distance_between_systems = config.getfloat(
            'Systems', 'Minimum distance'
        )

        # The gaussian mean and standard deviation params for number of
        # stars per system
        self.__stars_per_system_mean = None
        self.stars_per_system_mean = config.getfloat(
            'Systems', 'Stars per system mean'
        )
        self.__stars_per_system_std = None
        self.stars_per_system_std = config.getfloat(
            'Systems', 'Stars per system std'
        )

        # The gaussian mean and standard deviation params for number of
        # planets per system
        self.__planets_per_system_mean = None
        self.planets_per_system_mean = config.getfloat(
            'Systems', 'Planets per system mean'
        )
        self.__planets_per_system_std = None
        self.planets_per_system_std = config.getfloat(
            'Systems', 'Planets per system mean'
        )

        # The gaussian mean and standard deviation params for number of
        # moons per planet
        self.__moons_per_planet_mean = None
        self.moons_per_planet_mean = config.getfloat(
            'Systems', 'Moons per planet mean'
        )
        self.__moons_per_planet_std = None
        self.moons_per_planet_std = config.getfloat(
            'Systems', 'Moons per planet mean'
        )

    ####################################################################
    # Configuration
    #
    @property
    def minimum_distance_between_systems(self):
        return self.__min_dist_between_systems

    @minimum_distance_between_systems.setter
    def minimum_distance_between_systems(self, value):
        self.__min_dist_between_systems = value * u.parsec

    @property
    def stars_per_system_mean(self):
        return self.__stars_per_system_mean

    @stars_per_system_mean.setter
    def stars_per_system_mean(self, value):
        self.__stars_per_system_mean = value

    @property
    def stars_per_system_std(self):
        return self.__stars_per_system_std

    @stars_per_system_std.setter
    def stars_per_system_std(self, value):
        self.__stars_per_system_std = value

    @property
    def planets_per_system_mean(self):
        return self.__planets_per_system_mean

    @planets_per_system_mean.setter
    def planets_per_system_mean(self, value):
        self.__planets_per_system_mean = value

    @property
    def planets_per_system_std(self):
        return self.__planets_per_system_std

    @planets_per_system_std.setter
    def planets_per_system_std(self, value):
        self.__planets_per_system_std = value

    @property
    def moons_per_planet_mean(self):
        return self.__moons_per_planet_mean

    @moons_per_planet_mean.setter
    def moons_per_planet_mean(self, value):
        self.__moons_per_planet_mean = value

    @property
    def moons_per_planet_std(self):
        return self.__moons_per_planet_std

    @moons_per_planet_std.setter
    def moons_per_planet_std(self, value):
        self.__moons_per_planet_std = value

    ####################################################################
    # Generators
    #
    def elliptical(self, *, num_systems, width, height, depth):
        return Galaxy(systems=[
            self.__build_system(position=position)
            for position in self.__elliptical_positions(
                num_systems, width, height, depth
            )
        ])

    def __elliptical_positions(self, n, width, height, depth):
        positions = []
        while len(positions) < n:
            # We generate randomly points in a circle of radius 1
            dist = random.random()
            phi = random.uniform(0, 2 * math.pi)
            x = math.sqrt(dist) * math.cos(phi)
            y = math.sqrt(dist) * math.sin(phi)
            # Then, compute the z coordinate randomly withing bounds
            z = self.__compute_galaxy_depth(dist, 1, 1) * random.uniform(-1, 1)
            # Finally, scale to the ellipse dimensions
            x = x * width / 2.0
            y = y * depth / 2.0
            z = z * height / 2.0
            # Append this position only if there is room for it
            positions.append(
                # FIXME checking like this is too slow
                # self.__check_systems_proximity(
                # position=
                SkyCoord(
                    x=x, y=y, z=z,
                    unit=u.parsec,
                    representation_type='cartesian'
                )  # ,
                # rest_of_positions=positions
                # )
            )
        return positions

    '''
    def spiral(self, *, num_systems, width, height, depth):
        return Galaxy(systems=[
            self.__build_system(position)
            for position in self.__spiral_positions(
                num_systems, width, height, depth
            )
        ])

    def __spiral_positions(
            self,
            systems_in_core,
            systems_in_arms,
            height,
            core_radius,
            total_radius,
            num_arms,
            arm_rotation,
            arm_width
    ):
        # Separation (in degrees) for each galaxy arm
        omega = (360.0 / num_arms) if num_arms > 0 else 0

        positions = []
        # First, generate the core positions
        while len(positions) < systems_in_core:
            # We generate randomly points in a circle of radius 1
            r = float('inf')
            while r > 1:
                r = random.expovariate(1)
            phi = random.uniform(0, 2 * math.pi)
            x = math.sqrt(r) * math.cos(phi)
            y = math.sqrt(r) * math.sin(phi)
            # Then, scale to the core dimension
            x = x * core_radius
            y = y * core_radius
            dist = math.sqrt(x ** 2 + y ** 2)
            z = self.__compute_galaxy_depth(dist, total_radius, height)
            z *= random.uniform(-1, 1)
            # Append this position only if there is room for it
            try:
                positions.append(
                    self.__check_systems_proximity(
                        position=SkyCoord(
                            x=x, y=y, z=z,
                            unit=u.parsec,
                            representation_type='cartesian'
                        ),
                        rest_of_positions=positions
                    )
                )
            except ValueError as e:
                print('WARN: Coordinate too close')

        # Now, generate the arms positions
        while len(positions) < systems_in_core + systems_in_arms:
            # Generate a distance in the arms area
            dist = core_radius + random.random() * (total_radius - core_radius)
            # Rotate to one of the arms and squeeze it based on the distance
            theta = (
                    (720.0 * arm_rotation * (dist / total_radius))
                    + random.gauss(0, 0.7) * arm_width
                    + omega * random.randrange(0, num_arms)
            )
            # Now, convert it to cartesian coordinates
            x = math.cos(theta * math.pi / 180.0) * dist
            y = math.sin(theta * math.pi / 180.0) * dist
            dist = math.sqrt(x ** 2 + y ** 2)
            z = self.__compute_galaxy_depth(dist, total_radius, height)
            z *= random.uniform(-1, 1)
            # Append this position only if there is room for it
            try:
                positions.append(
                    self.__check_systems_proximity(
                        position=SkyCoord(
                            x=x, y=y, z=z,
                            unit=u.parsec,
                            representation_type='cartesian'
                        ),
                        rest_of_positions=positions
                    )
                )
            except ValueError as e:
                print('WARN: Coordinate too close')

        return positions
    '''

    ####################################################################
    # Utility functions
    #
    def __check_systems_proximity(self, position, rest_of_positions):
        """Ensures no two coordinates are too close.

        The method will return the original coordinate silently in case
        it's located far enough of the rest of coordinates.

        :param position: The position to check against the rest of
            positions.
        :param rest_of_positions: The rest of positions.
        :return: The original coordinate.
        :raise ValueError: In case the coordinate is too close to the
            rest of positions
        """
        for other_coordinate in rest_of_positions:
            dist = position.separation_3d(other_coordinate)
            if dist < self.minimum_distance_between_systems:
                raise ValueError('System to close to another')
        return position

    @staticmethod
    def __compute_galaxy_depth(x, max_distance, galaxy_height):
        """Modified generic logistic function."""
        tau = (max_distance / 2 - x) / 100
        return galaxy_height + -galaxy_height / (1 + math.exp(tau))

    def __build_system(self, *, position):
        """Creates a random system in the specified position.

        :param position: The position of the system in the galaxy.
        :return: A newly created system.
        """
        # First, let's see how many stars exists in out system (there
        # should be at least one star per system, hence the max op)
        num_stars = max(1, round(random.gauss(
            mu=self.stars_per_system_mean,
            sigma=self.stars_per_system_std,
        )))
        stars = [self.star_builder.build() for _ in range(num_stars)]

        # Then how many planets, if any. This time, the minimum number
        # of planets a system can hold is zero (just a Stellar System)
        num_planets = max(0, round(random.gauss(
            mu=self.planets_per_system_mean,
            sigma=self.planets_per_system_std,
        )))
        planets = [self.__build_planet() for _ in range(num_planets)]

        # And now, create the system
        return System(
            position=position,
            stars=stars,
            planets=planets,
        )

    def __build_planet(self):
        """Generates a random planet.

        The planet will be generated with n moons, being n a value
        chosen from a normal distribution with the mean and standard
        deviation specified by the arguments. That value will be rounded
        to the nearest integer and it will be never lower than 0.

        :return: A newly created planet.
        """
        # First, let's see how many moons belong to the planet, if any.
        # The minimum number of moons per planet is 0, so that's the
        # reason behind the max operator.
        num_moons = max(0, round(random.gauss(
            mu=self.moons_per_planet_mean,
            sigma=self.moons_per_planet_std,
        )))
        moons = [self.__build_moon() for _ in range(num_moons)]

        # And now, create the planet
        return Planet(moons=moons)

    @staticmethod
    def __build_moon():
        """Creates a random moon.

        :return: A newly created moon.
        """
        return Moon()
