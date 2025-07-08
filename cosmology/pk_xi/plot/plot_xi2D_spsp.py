import copy
import numpy as np
import matplotlib.pyplot as plt

cmap = "jet"
xis2_cnt_levels = np.linspace(-200, 200, 101)

outfig = False

prefix = "DDspsp_uni2"
filename = prefix + ".dat"
outname_prefix = prefix

data = np.loadtxt(filename, unpack=True)

sperp = np.unique(data[0])
spara = np.unique(data[1])

if not np.isclose(np.abs(sperp[0]), np.abs(sperp[-1])):
    sperp -= sperp[0]

if not np.isclose(np.abs(spara[0]), np.abs(spara[-1])):
    spara -= spara[0]

nsperp = len(sperp)
nspara = len(spara)

xi = data[2].reshape((nsperp, nspara))
dd = copy.deepcopy(data[3].reshape((nsperp, nspara)))

grid_s_perp, grid_s_para = np.meshgrid(sperp, spara, indexing='ij')

ss = grid_s_perp**2 + grid_s_para**2
xis2 = xi * ss

if spara.min() >= 0:
    grid_s_perp = np.concatenate([-np.fliplr(grid_s_perp), grid_s_perp], axis=1)
    grid_s_para = np.concatenate([np.fliplr(grid_s_para), grid_s_para], axis=1)
    xis2 = np.concatenate([np.fliplr(xis2), xis2], axis=1)

    grid_s_perp = np.concatenate([np.flipud(grid_s_perp), grid_s_perp], axis=0)
    grid_s_para = np.concatenate([-np.flipud(grid_s_para), grid_s_para], axis=0)
    xis2 = np.concatenate([np.flipud(xis2), xis2], axis=0)
else:
    grid_s_perp = np.concatenate([-np.fliplr(grid_s_perp), grid_s_perp], axis=1)
    grid_s_para = np.concatenate([np.fliplr(grid_s_para), grid_s_para], axis=1)
    xis2 = np.concatenate([np.fliplr(xis2), xis2], axis=1)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

im = ax.contourf(grid_s_perp, grid_s_para, xis2, levels=xis2_cnt_levels, cmap=cmap, extend='both')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'$s^2 \xi(s_\perp, s_\parallel)$', fontsize="x-large")
ax.set_xlabel(r'$s_\perp$ [Mpc/h]', fontsize="x-large")
ax.set_ylabel(r'$s_\parallel$ [Mpc/h]', fontsize="x-large")
ax.set_title("Redshift-Space Distortions", fontsize="xx-large")
ax.set_aspect('equal', adjustable='box')

if outfig: plt.savefig(outname_prefix + ".png", bbox_inches="tight", dpi=150)
plt.show()
