import numpy as np
from datetime import datetime, timedelta

from sgp4.api import Satrec, WGS72
from sgp4.conveniences import jday_datetime

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, ITRS, CartesianRepresentation, EarthLocation

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# on mac:
# brew install gcc; export FC=gfortran ; export F77=gfortran ; pip install --no-cache-dir iri2016 ; brew install cmake
# on linux, it should just be a simple pip install iri2016
from iri2016 import IRI

def plot_orbit(altitude=450, n_orbits=5, epoch = datetime(2006, 2, 25, 0, 0, 0)):
    # only historical dates with iri2016, as it relies on a table of solar radio flux and geomagnetic activity index.
    # look into using pyiri with a user defined f10.7 cm radio flux and ap index to manually explore different solar activity levels.
    # I haven't figured out how to feed f10.7 and ap to iri2016.
    
    mu = 398600.4418        # km^3/s^2
    Re = 6378.137           # km

    a = Re + altitude       # semi-major axis [km]
    e = 0.0

    inclination_deg = 54.0
    raan_deg = 40.0
    argp_deg = 0.0
    mean_anomaly_deg = 0.0

    inclination = np.deg2rad(inclination_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    mean_anomaly = np.deg2rad(mean_anomaly_deg)

    # Mean motion (rad/min required by SGP4)
    n_rad_s = np.sqrt(mu / a**3)
    n_rad_min = n_rad_s * 60.0

    
    jd, fr = jday_datetime(epoch)
    epoch_jd = jd + fr


    # ============================================================
    # 2. Initialize SGP4 from Keplerian elements
    # ============================================================

    sat = Satrec()

    sat.sgp4init(
        WGS72,        # gravity model
        'i',          # improved mode
        99999,        # satellite number
        epoch_jd - 2433281.5,  # epoch in days since 1949-12-31
        0.0,          # bstar drag term
        0.0,          # ndot
        0.0,          # nddot
        e,            # eccentricity
        argp,         # argument of perigee (rad)
        inclination,  # inclination (rad)
        mean_anomaly, # mean anomaly (rad)
        n_rad_min,    # mean motion (rad/min)
        raan           # RAAN (rad)
    )
    period = n_orbits*2 * np.pi * np.sqrt(a**3 / mu)
    num_points = n_orbits*100

    times = [epoch + timedelta(seconds=s)
            for s in np.linspace(0, period, num_points)]


    electron_density = []
    lats=[]
    lons=[]
    for t in times:

        jd, fr = jday_datetime(t)
        error, r_teme, v_teme = sat.sgp4(jd, fr)
        if error != 0:
            electron_density.append(np.nan)
            continue

        teme = TEME(
            CartesianRepresentation(r_teme * u.km),
            obstime=Time(t)
        )

        itrs = teme.transform_to(ITRS(obstime=Time(t)))
        location = EarthLocation(
            x=itrs.x,
            y=itrs.y,
            z=itrs.z
        )

        lat = location.lat.deg
        lon = location.lon.deg
        alt_km = location.height.to(u.km).value
        alt_range = [alt_km, alt_km, 1]
        iri_out = IRI(t, alt_range, lat, lon)
        ne = iri_out["ne"][0]   # m^-3
        electron_density.append(ne)
        lats.append(lat)
        lons.append(lon)


    electron_density = np.array(electron_density)
    plasma_freq_MHz=9*np.sqrt(electron_density)/1e6



    # Create figure
    plt.figure(figsize=(12,5))

    # -------------------------
    # 1. Map of plasma frequency vs lat/lon
    # -------------------------
    plt.subplot(1,2,1)
    sc = plt.scatter(lons, lats, c=plasma_freq_MHz, s=20, cmap='viridis')
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("In-situ plasma frequency map %d km"%(altitude))
    cb = plt.colorbar(sc)
    cb.set_label("Plasma frequency (MHz)")

    # -------------------------
    # 2. Plasma frequency vs time
    # -------------------------
    date_str = times[0].strftime('%Y-%m-%d')
    plt.subplot(1,2,2)
    plt.plot(times, plasma_freq_MHz, '.', markersize=6)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Plasma frequency (MHz)")
    # Format x-axis as readable UTC
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()  # rotate labels
    plt.title(f"Plasma frequency along orbit ({date_str})")
    plt.tight_layout()
    plt.show()


plot_orbit(altitude=450, n_orbits=5, epoch = datetime(2020, 2, 25, 0, 0, 0))
plot_orbit(altitude=550, n_orbits=5, epoch = datetime(2020, 2, 25, 0, 0, 0))
plot_orbit(altitude=650, n_orbits=5, epoch = datetime(2020, 2, 25, 0, 0, 0))
