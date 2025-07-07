import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

xis2_cnt_levels = np.linspace(-20, 20, 101)

outfig = True

interp = True
# interp = False

prefix = "DDsmu_uni1"
filename = prefix + ".dat"
outname_prefix = prefix


data = np.loadtxt(filename, unpack=True)

scen = np.unique(data[0])
mucen = np.unique(data[1])

ns = len(scen)
nmu = len(mucen)

opposite = np.isclose(np.abs(mucen[0]), np.abs(mucen[-1]))
print("is_opposite : ", opposite)

xi = data[2].reshape((ns, nmu))
dd = copy.deepcopy(data[3].reshape((ns, nmu)))

xis2 = xi * scen[:, None]**2

s, mu = np.meshgrid(scen, mucen, indexing='ij')
s1 = s * np.sqrt(1 - mu**2)
s2 = s * mu

if interp:
    xis2 = xis2.flatten()
    s1 = s1.flatten()
    s2 = s2.flatten()
    npts = 200
    grid_x, grid_y = np.meshgrid(
        np.linspace(s1.min(), s1.max(), npts),
        np.linspace(s2.min(), s2.max(), npts)
    )

    xis2 = griddata((s1, s2), xis2, (grid_x, grid_y),
                    method='nearest')

    s1 = grid_x
    s2 = grid_y

# Flip vertically
s1 = np.concatenate([-np.flipud(s1), s1], axis=0)
s2 = np.concatenate([np.flipud(s2), s2], axis=0)
xis2 = np.concatenate([np.flipud(xis2), xis2], axis=0)

if not opposite:
    # Flip horizontally
    s1 = np.concatenate([np.fliplr(s1), s1], axis=1)
    s2 = np.concatenate([-np.fliplr(s2), s2], axis=1)
    xis2 = np.concatenate([np.fliplr(xis2), xis2], axis=1)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])

# xis2_cnt_levels = np.linspace(-xis2.max(), xis2.max(), 101)
im = ax.contourf(s1, s2, xis2, levels=xis2_cnt_levels, cmap="jet")

plt.colorbar(im, ax=ax, label=r'$\xi(s,\mu)\,s^2$')
ax.set_xlabel(r'$s_\perp$ [Mpc/h]', fontsize="xx-large")
ax.set_ylabel(r'$s_\parallel$ [Mpc/h]', fontsize="xx-large")
plt.title(r'$\xi(s_\perp, s_\parallel)\,s^2$', fontsize="xx-large")
if outfig: plt.savefig(outname_prefix + "_xi2D.png", bbox_inches="tight")
plt.show()


s, mu = np.meshgrid(scen, mucen, indexing='ij')
s_edges = np.concatenate(([scen[0] - (scen[1] - scen[0]) / 2],
                          (scen[:-1] + scen[1:]) / 2,
                          [scen[-1] + (scen[-1] - scen[-2]) / 2]))
mu_edges = np.concatenate(([mucen[0] - (mucen[1] - mucen[0]) / 2],
                           (mucen[:-1] + mucen[1:]) / 2,
                           [mucen[-1] + (mucen[-1] - mucen[-2]) / 2]))

"""
ds = np.diff(s_edges)[:, None]     # shape (ns,1)
dmu = np.diff(mu_edges)[None, :]    # shape (1,nmu)

V_shell = 4 * np.pi / 3 * (s_edges[1:]**3 - s_edges[:-1]**3)
dV_sm = V_shell[:, None] * dmu / 2

dd = copy.deepcopy(data[3].reshape((ns, nmu)))
dd = dd / dV_sm

s2 = s * mu
s1 = s * np.sqrt(1 - mu**2)

if interp:
    dd = dd.flatten()
    s1 = s1.flatten()
    s2 = s2.flatten()
    npts = 200
    grid_x, grid_y = np.meshgrid(
        np.linspace(s1.min(), s1.max(), npts),
        np.linspace(s2.min(), s2.max(), npts)
    )

    dd = griddata((s1, s2), dd, (grid_x, grid_y),
                  method='nearest')

    s1 = grid_x
    s2 = grid_y

# Flip vertically
s1 = np.concatenate([-np.flipud(s1), s1], axis=0)
s2 = np.concatenate([np.flipud(s2), s2], axis=0)
dd = np.concatenate([np.flipud(dd), dd], axis=0)

if not opposite:
    # Flip horizontally
    s1 = np.concatenate([np.fliplr(s1), s1], axis=1)
    s2 = np.concatenate([-np.fliplr(s2), s2], axis=1)
    dd = np.concatenate([np.fliplr(dd), dd], axis=1)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])

dd_cnt_levels = np.linspace(1, dd.max(), 101)

# im = ax.contourf(s1, s2, np.fliplr(dd) + dd, levels=dd_cnt_levels, cmap="jet")
# im = ax.contourf(s1, s2, dd, levels=dd_cnt_levels, cmap="jet")
im = ax.contourf(s1, s2, dd, cmap="jet", levels=101)

plt.colorbar(im, ax=ax, label=r'$DD$')
ax.set_xlabel(r'$s_\perp$ [Mpc/h]', fontsize="xx-large")
ax.set_ylabel(r'$s_\parallel$ [Mpc/h]', fontsize="xx-large")
plt.title(r'$DD$', fontsize="xx-large")
plt.savefig(outname_prefix + "_DD2D_norm.png", bbox_inches="tight")
plt.show()
"""

dd = copy.deepcopy(data[3].reshape((ns, nmu)))
s, mu = np.meshgrid(scen, mucen, indexing='ij')
s2 = s * mu
s1 = s * np.sqrt(1 - mu**2)

if interp:
    dd = dd.flatten()
    s1 = s1.flatten()
    s2 = s2.flatten()
    npts = 200
    grid_x, grid_y = np.meshgrid(
        np.linspace(s1.min(), s1.max(), npts),
        np.linspace(s2.min(), s2.max(), npts)
    )

    dd = griddata((s1, s2), dd, (grid_x, grid_y),
                  method='nearest')

    s1 = grid_x
    s2 = grid_y

# Flip vertically
s1 = np.concatenate([-np.flipud(s1), s1], axis=0)
s2 = np.concatenate([np.flipud(s2), s2], axis=0)
dd = np.concatenate([np.flipud(dd), dd], axis=0)

if not opposite:
    # Flip horizontally
    s1 = np.concatenate([np.fliplr(s1), s1], axis=1)
    s2 = np.concatenate([-np.fliplr(s2), s2], axis=1)
    dd = np.concatenate([np.fliplr(dd), dd], axis=1)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])

dd_cnt_levels = np.linspace(1, dd.max(), 101)

# im = ax.contourf(s1, s2, np.fliplr(dd) + dd, levels=dd_cnt_levels, cmap="jet")
# im = ax.contourf(s1, s2, dd, levels=dd_cnt_levels, cmap="jet")
im = ax.contourf(s1, s2, dd, cmap="jet", levels=101)

plt.colorbar(im, ax=ax, label=r'$DD$')
ax.set_xlabel(r'$s_\perp$ [Mpc/h]', fontsize="xx-large")
ax.set_ylabel(r'$s_\parallel$ [Mpc/h]', fontsize="xx-large")
plt.title(r'$DD$', fontsize="xx-large")
if outfig: plt.savefig(outname_prefix + "_DD2D.png", bbox_inches="tight")
plt.show()
