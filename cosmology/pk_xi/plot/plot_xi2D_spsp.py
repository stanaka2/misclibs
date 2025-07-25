import copy
import numpy as np
import matplotlib.pyplot as plt

cmap = "jet"
xi_cnt_levels = np.linspace(-200, 200, 101)

outfig = False

prefix = "./test1"
filename = prefix + ".dat"
outname_prefix = prefix

data = np.loadtxt(filename, unpack=True)

sperp = np.unique(data[0])
spara = np.unique(data[1])

nsperp = len(sperp)
nspara = len(spara)

xi = data[2].reshape((nsperp, nspara))
dd = copy.deepcopy(data[3].reshape((nsperp, nspara)))

grid_s_perp, grid_s_para = np.meshgrid(sperp, spara, indexing='ij')

ss = grid_s_perp**2 + grid_s_para**2
xi = np.log10(xi)

grid_s_perp = np.concatenate([-np.flipud(grid_s_perp), grid_s_perp], axis=0)
grid_s_para = np.concatenate([np.flipud(grid_s_para), grid_s_para], axis=0)
xi = np.concatenate([np.flipud(xi), xi], axis=0)

if spara.min() >= 0:
    grid_s_perp = np.concatenate([np.fliplr(grid_s_perp), grid_s_perp], axis=1)
    grid_s_para = np.concatenate([-np.fliplr(grid_s_para), grid_s_para], axis=1)
    xi = np.concatenate([np.fliplr(xi), xi], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

X = np.linspace(grid_s_perp.min(), grid_s_perp.max(), xi.shape[0])
Y = np.linspace(grid_s_para.min(), grid_s_para.max(), xi.shape[1])
XX, YY = np.meshgrid(X, Y, indexing='ij')

vmin = -2
vmax = 1
levels = np.linspace(vmin, vmax, 11)

if 0:
    im = ax.contourf(XX, YY, xi, cmap=cmap, levels=levels, extend="both")
elif 0:
    im = ax.pcolormesh(XX, YY, xi, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.contour(XX, YY, xi, levels=levels, colors='black')
else:
    extent = [grid_s_perp.min(), grid_s_perp.max(), grid_s_para.min(), grid_s_para.max()]
    im = ax.imshow(xi.T, origin='lower', extent=extent, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    ax.contour(XX, YY, xi, levels=levels, colors='black')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'$s^2 \xi(s_\perp, s_\parallel)$', fontsize="x-large")
ax.set_xlabel(r'$s_\perp$ [Mpc/h]', fontsize="x-large")
ax.set_ylabel(r'$s_\parallel$ [Mpc/h]', fontsize="x-large")
ax.set_title("Redshift-Space Distortions", fontsize="xx-large")
ax.set_aspect('equal', adjustable='box')

if outfig: plt.savefig(outname_prefix + ".png", bbox_inches="tight", dpi=150)
plt.show()
