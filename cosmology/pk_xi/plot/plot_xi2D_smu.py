import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

cmap = "jet"
xis2_cnt_levels = np.linspace(-200, 200, 101)

outfig = False

prefix = "DDsmu_uni1"
filename = prefix + ".dat"
outname_prefix = prefix

data = np.loadtxt(filename, unpack=True, comments='#')

scen = np.unique(data[0])
mucen = np.unique(data[1])
ns = len(scen)
nmu = len(mucen)
xi = data[2].reshape((ns, nmu))
s, mu = np.meshgrid(scen, mucen, indexing='ij')

s_perp = s * np.sqrt(1 - mu**2)
s_para = s * mu

xis2 = xi * s**2
weight = s / np.sqrt(s**2 - s_para**2 + 1e-10)
xis2 *= weight

s_perp_flat = s_perp.flatten()
s_para_flat = s_para.flatten()
xis2_flat = xis2.flatten()
npts = 200
grid_s_perp, grid_s_para = np.meshgrid(
    np.linspace(s_perp_flat.min(), s_perp_flat.max(), npts),
    np.linspace(s_para_flat.min(), s_para_flat.max(), npts)
)

xis2_grid = griddata((s_perp_flat, s_para_flat), xis2_flat,
                     (grid_s_perp, grid_s_para), method='linear')

nan_mask = np.isnan(xis2_grid)
if np.any(nan_mask):
    xis2_filled = griddata((s_perp_flat, s_para_flat), xis2_flat,
                           (grid_s_perp[nan_mask], grid_s_para[nan_mask]),
                           method='nearest')
    xis2_grid[nan_mask] = xis2_filled

if s_para.min() >= 0:
    s_perp = np.concatenate([-np.fliplr(grid_s_perp), grid_s_perp], axis=1)
    s_para = np.concatenate([np.fliplr(grid_s_para), grid_s_para], axis=1)
    xis2 = np.concatenate([np.fliplr(xis2_grid), xis2_grid], axis=1)

    s_perp = np.concatenate([np.flipud(s_perp), s_perp], axis=0)
    s_para = np.concatenate([-np.flipud(s_para), s_para], axis=0)
    xis2 = np.concatenate([np.flipud(xis2), xis2], axis=0)
else:
    s_perp = np.concatenate([-np.fliplr(grid_s_perp), grid_s_perp], axis=1)
    s_para = np.concatenate([np.fliplr(grid_s_para), grid_s_para], axis=1)
    xis2 = np.concatenate([np.fliplr(xis2_grid), xis2_grid], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

im = ax.contourf(s_perp, s_para, xis2, levels=xis2_cnt_levels, cmap=cmap, extend='both')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'$s^2 \xi(s_\perp, s_\parallel)$', fontsize="x-large")
ax.set_xlabel(r'$s_\perp$ [Mpc/h]', fontsize="x-large")
ax.set_ylabel(r'$s_\parallel$ [Mpc/h]', fontsize="x-large")
ax.set_title("Redshift-Space Distortions", fontsize="xx-large")
ax.set_aspect('equal', adjustable='box')

if outfig: plt.savefig(outname_prefix + ".png", bbox_inches="tight", dpi=150)
plt.show()
