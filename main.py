from config import config
from imperia.galaxy import GalaxyBuilder

if __name__ == '__main__':
    galaxy_builder = GalaxyBuilder(config=config)
    galaxy = galaxy_builder.elliptical(
        num_systems=10 ** 3,
        width=2.191 * 10 ** 19,
        height=7.097 * 10 ** 18,
        depth=1.419 * 10 ** 20,
        # num_systems=10 ** 5,
        # width=2.191 * 10 ** 22,
        # height=7.097 * 10 ** 20,
        # depth=1.419 * 10 ** 22,
    )
    print(str(galaxy))
