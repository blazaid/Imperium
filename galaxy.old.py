import abc
import math
import random


class System(metaclass=abc.ABCMeta):
    """Any possible system in the galaxy."""

    def __init__(self, x: float, y: float, z: float):
        """Initializes this object.

        :param x: The x coordinate in the galaxy in parsecs.
        :param y: The y coordinate in the galaxy in parsecs.
        :param x: The z coordinate in the galaxy in parsecs.
        """
        self._x, self._y, self._z = x, y, z

    @property
    def x(self):
        return self._y

    @property
    def y(self):
        return self._x

    @property
    def z(self):
        return self._z

    @abc.abstractmethod
    def color(self):
        """System's color emission.

        Depends on the composition, thus the subclasses.

        :return: A 4-tuple RGBA where each value is a value in the
            [0, 1] interval.
        """


class StellarSystem(System):
    def __init__(self, *, x, y, z, stars):
        """Initializes this object.

        :param x: The x coordinate in the galaxy in parsecs.
        :param y: The y coordinate in the galaxy in parsecs.
        :param x: The z coordinate in the galaxy in parsecs.
        """
        super().__init__(x, y, z)
        self.stars = stars

    def color(self):
        max(self.stars, key=lambda star: star.mass).color()

    def __str__(self):
        return str([str(star) for star in self.stars])


class Star:
    def __init__(self, *, mass):
        """Initializes this object.

        :param mass: The mass of this star in M☉ (solar masses).
        """
        self._mass = mass

    @property
    def max(self):
        return self._mass

    def color(self):
        pass

    def __str__(self):
        return 'star'


class PlanetarySystem(System):
    def __init__(self, *, x, y, z, stellar_system, planets):
        super().__init__(x, y, z)
        self.stellar_system = stellar_system
        self.planets = planets

    def color(self):
        """A planetary system has the color of its stellar system."""
        return self.stellar_system.color()

    def __str__(self):
        return f'{str(self.stellar_system)}\n\t' + str([str(planet) for planet in self.planets])


class Planet:
    def __str__(self):
        return 'planet'


default_galaxy_config = {
    'systems_proportions': {
        'stellar': 1,
        'planetary': 2
    },
    'stars': {
        'per_system': {
            'n': [1, 2, 3],
            'w': [1000, 10, 1],
        },
    },
    'planets': {
        'per_system': {
            'n': [1, 2, 3],
            'w': [1000, 10, 1],
        },
    },
    'planets_in_planetary_system': {
        'n': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'w': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    },
}


class Galaxy:
    """The place where the action of the game take place."""

    def __init__(self, size_x, size_y, size_z, config=None):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z

        self.systems = []

    def append(self, system):
        self.systems.append(system)

    def __iter__(self):
        return self.systems.__iter__()

    def __str__(self):
        return '\n'.join(str(system) for system in self.systems)


class GalaxyBuilder:
    def __init__(self, *, config=None):
        self.config = config or default_galaxy_config

    def elliptical(self, *, num_systems, width, height, depth):
        return [
            self.random_system(x=x, y=y, z=z)
            for x, y, z in self.__elliptical_positions(
                num_systems, width, height, depth
            )
        ]

    def __elliptical_positions(self, n, width, height, depth):
        positions = []
        for _ in range(n):
            # We generate randomly points in a circle of radius 1
            dist = random.random()
            phi = random.uniform(0, 2 * math.pi)
            x = math.sqrt(dist) * math.cos(phi)
            y = math.sqrt(dist) * math.sin(phi)
            # Then, compute the z coordinate randomly withing bounds
            z = self.__compute_system_depth(dist, 1, 1) * random.uniform(-1, 1)
            # Finally, scale to the ellipse dimensions
            x = x * width / 2.0
            y = y * depth / 2.0
            z = z * height / 2.0
            positions.append((x, y, z))
        return positions

    @staticmethod
    def __compute_system_depth(x, max_distance, galaxy_height):
        """Modified generic logistic function."""
        return galaxy_height + -galaxy_height / (1 + math.exp((max_distance / 2 - x) / 100))

    def construct_spiral(self, *, systems_in_core, systems_in_arms, height, core_radius, total_radius, num_arms,
                         arm_rotation,
                         arm_width):
        galaxy = Galaxy(total_radius * 2, total_radius * 2, height)

        # Separación (en grados) entre cada brazo
        # Separación (en grados) entre cada brazo
        omega = (360.0 / num_arms) if num_arms > 0 else 0

        # Generamos las estrellas del núcleo central
        for _ in range(systems_in_core):
            # Generamos aleatoriamente un punto en un círculo de radio 1
            r = float('inf')
            while r > 1:
                r = random.expovariate(1)
            phi = random.uniform(0, 2 * math.pi)
            x = math.sqrt(r) * math.cos(phi)
            y = math.sqrt(r) * math.sin(phi)
            # Escalamos a la dimensión del disco
            x = x * core_radius
            y = y * core_radius
            # Añadimos el nuevo sistema a la galaxia (ya calcularemos su coordenada z)
            galaxy.append(self.random_system(x=x, y=y, z=0))

        # Generamos las estrellas de los brazos
        for _ in range(systems_in_arms):
            # Generamos una distancia (dentro del anillo de la localización de los brazos)
            dist = core_radius + random.random() * (total_radius - core_radius)
            # Ponemos el sistema en uno de los brazos y lo rotamos en función de la distancia
            theta = (
                    (720.0 * arm_rotation * (dist / total_radius))
                    + random.gauss(0, 0.7) * arm_width
                    + omega * random.randrange(0, num_arms)
            )
            # Y lo pasamos a cartesianas
            x = math.cos(theta * math.pi / 180.0) * dist
            y = math.sin(theta * math.pi / 180.0) * dist
            # Añadimos el nuevo sistema a la galaxia (ya calcularemos su coordenada z)
            galaxy.append(self.random_system(x=x, y=y, z=0))

        for system in galaxy:
            dist = math.sqrt(system.x ** 2 + system.y ** 2)
            system.z = random.uniform(-1, 1) * GalaxyBuilder.compute_system_depth(dist, total_radius, height)

        return galaxy

    def random_system(self, *, x, y, z):
        """Creates a new system randomly.

        The parameters impact on the probability for the systems to be
        generated.

        :param x: The system's x coordinate.
        :param y: The system's y coordinate.
        :param z: The system's z coordinate.
        """

        def random_star():
            return Star(mass=1)

        def random_planet():
            return Planet()

        # For now, all the different systems (stellar and planetary)
        # contain stars, so we generate them first
        per_system = self.config['stars']['per_system']['n']
        weights = self.config['stars']['per_system']['w']
        num_stars = random.choices(per_system, weights=weights)[0]
        stars = [random_star() for _ in range(num_stars)]
        # Now, we choose randomly the kind of system to generate
        types, weights = zip(*self.config['systems_proportions'].items())
        system_type = random.choices(types, weights=weights)[0]
        if system_type == 'stellar':
            # Well, a stellar system is just that, some stars together
            return StellarSystem(x=x, y=y, z=z, stars=stars)
        elif system_type == 'planetary':
            # For a planetary system we need planets, so let's generate
            # some
            per_system = self.config['planets']['per_system']['n']
            weights = self.config['planets']['per_system']['w']
            num_planets = random.choices(per_system, weights=weights)[0]
            planets = [random_planet() for _ in range(num_planets)]
            return PlanetarySystem(
                x=x, y=y, z=z,
                stellar_system=StellarSystem(x=x, y=y, z=z, stars=stars),
                planets=planets
            )
        else:
            raise ValueError(f'Unknown type: {system_type}')


if __name__ == '__main__':
    galaxy = GalaxyBuilder().elliptical(
        num_systems=2000,
        width=1000,
        height=100,
        depth=500,
    )
    print(str(galaxy))
