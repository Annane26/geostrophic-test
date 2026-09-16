"""
Geostrophic wind around a low-pressure system.

Above the atmospheric boundary layer, wind results from a near-balance
between the pressure gradient force (pushing air from high to low
pressure) and the Coriolis force (deflecting moving air, to the right
of its motion in the Northern Hemisphere). The resulting "geostrophic
wind" blows parallel to the isobars (lines of constant pressure)
rather than across them:

    u_g = -(1 / (rho * f)) * dP/dy
    v_g =  (1 / (rho * f)) * dP/dx

where f = 2 * Omega * sin(latitude) is the Coriolis parameter.

This script builds a synthetic low-pressure system (like a weather map),
computes its geostrophic wind field, and checks two textbook facts:
  1. The wind blows perpendicular to the pressure gradient (i.e. along
     isobars, not across them).
  2. In the Northern Hemisphere, wind circulates counterclockwise
     around a low ("cyclonic" flow).
"""

import numpy as np
import matplotlib.pyplot as plt

Omega = 7.292e-5   # rad/s, Earth's rotation rate
rho = 1.2          # kg/m^3, air density
lat0 = 45.0        # degrees N
f = 2 * Omega * np.sin(np.radians(lat0))

L = 4.0e6          # domain size, 4000 km
n = 121
x = np.linspace(-L / 2, L / 2, n)
y = np.linspace(-L / 2, L / 2, n)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="ij")

P0 = 101325.0      # Pa, background pressure
A = 2000.0         # Pa, depth of the low (20 hPa)
sigma = L / 6.0

P = P0 - A * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

dPdx, dPdy = np.gradient(P, dx, dy)

u_g = -(1.0 / (rho * f)) * dPdy
v_g = (1.0 / (rho * f)) * dPdx

if __name__ == "__main__":
    plt.figure(figsize=(7, 6))
    cs = plt.contour(X / 1000, Y / 1000, P / 100, levels=12, cmap="coolwarm")
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.0f hPa")
    skip = 8
    plt.quiver(X[::skip, ::skip] / 1000, Y[::skip, ::skip] / 1000,
               u_g[::skip, ::skip], v_g[::skip, ::skip], color="black")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title(f"Geostrophic wind around a low ({lat0:.0f}N)")
    plt.gca().set_aspect("equal")
    plt.savefig("geostrophic_wind_map.png")
    print("Saved geostrophic_wind_map.png")

    grad_mag = np.sqrt(dPdx**2 + dPdy**2)
    wind_mag = np.sqrt(u_g**2 + v_g**2)
    valid = grad_mag > 1e-6

    cos_angle = (u_g[valid] * dPdx[valid] + v_g[valid] * dPdy[valid]) / (wind_mag[valid] * grad_mag[valid])
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))

    deviation = angle_deg - 90.0
    mean_deviation = np.mean(np.abs(deviation))
    pad = max(np.max(np.abs(deviation)) * 2, 1e-9)

    plt.figure(figsize=(7, 5))
    plt.hist(deviation, bins=40, range=(-pad, pad), color="steelblue")
    plt.axvline(0, color="red", linestyle="--", label="0 deviation (perfectly perpendicular)")
    plt.xlabel("Deviation from 90 degrees between wind and pressure gradient")
    plt.ylabel("Number of grid points")
    plt.title("Geostrophic wind should be perpendicular to the pressure gradient")
    plt.legend()
    plt.savefig("geostrophic_angle_check.png")
    print("Saved geostrophic_angle_check.png")

    print(f"Mean deviation from 90 degrees: {mean_deviation:.2e} degrees")
    assert mean_deviation < 1.0, "Geostrophic wind is not perpendicular to the pressure gradient!"

    i_center = n // 2
    j_north = i_center + n // 4
    j_east_i = i_center + n // 4

    u_north = u_g[i_center, j_north]
    v_east = v_g[j_east_i, i_center]

    print(f"u_g north of the low: {u_north:.2f} m/s (expect negative -> westward)")
    print(f"v_g east of the low:  {v_east:.2f} m/s (expect positive -> northward)")
    assert u_north < 0, "Expected westward wind north of the low (cyclonic rotation)!"
    assert v_east > 0, "Expected northward wind east of the low (cyclonic rotation)!"
    print("PASS: wind is perpendicular to the pressure gradient and circulates "
          "counterclockwise (cyclonically) around the Northern Hemisphere low.")
