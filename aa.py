from astropy import units as u
from astropy.coordinates import SkyCoord

c1 = SkyCoord(x=1, y=2, z=3, unit=u.parsec, representation_type='cartesian')
c2 = SkyCoord(x=2, y=4, z=1, unit=u.parsec, representation_type='cartesian')
print(c1.separation_3d(c2))
