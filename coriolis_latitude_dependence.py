"""
How geostrophic wind speed depends on latitude.

For the SAME pressure gradient force, geostrophic wind speed is:

    Vg = (1 / (rho * f)) * (dP/dn),   f = 2 * Omega * sin(latitude)

Since f shrinks toward the equator, the same pressure gradient
produces much STRONGER geostrophic wind at low latitudes -- and the
formula breaks down entirely at the equator (f = 0), which is why
the geostrophic approximation is not used there.
"""

import numpy as np
import matplotlib.pyplot as plt

Omega = 7.292e-5
rho = 1.2
dPdn = 3.0e-3   # Pa/m, a typical synoptic-scale pressure gradient (~3 hPa / 100 km)

def geostrophic_speed(lat_deg):
    f = 2 * Omega * np.sin(np.radians(lat_deg))
    return dPdn / (rho * f)

if __name__ == "__main__":
    lat = np.linspace(5, 85, 200)
    Vg = geostrophic_speed(lat)

    plt.figure(figsize=(8, 5))
    plt.plot(lat, Vg, color="darkgreen")
    plt.xlabel("Latitude (degrees N)")
    plt.ylabel("Geostrophic wind speed (m/s)")
    plt.title("Same pressure gradient, different latitude: geostrophic wind speed")
    plt.axvline(30, color="gray", linestyle=":", label="30N (typical jet stream latitude range starts)")
    plt.legend()
    plt.savefig("geostrophic_latitude_dependence.png")
    print("Saved geostrophic_latitude_dependence.png")

    print(f"Geostrophic wind speed at 10N: {geostrophic_speed(10):.1f} m/s")
    print(f"Geostrophic wind speed at 45N: {geostrophic_speed(45):.1f} m/s")
    print(f"Geostrophic wind speed at 80N: {geostrophic_speed(80):.1f} m/s")

    constant_check = Vg * np.sin(np.radians(lat))
    rel_spread = (constant_check.max() - constant_check.min()) / constant_check.mean()
    print(f"Relative spread of Vg*sin(lat) across all latitudes: {rel_spread:.2e}")
    assert rel_spread < 1e-6, "Vg does not scale as 1/sin(latitude) as expected!"
    assert np.all(np.diff(Vg) < 0), "Expected Vg to decrease monotonically with latitude!"
    print("PASS: geostrophic wind speed scales as 1/sin(latitude), confirming "
          "the same pressure gradient drives much stronger winds at low latitudes.")
