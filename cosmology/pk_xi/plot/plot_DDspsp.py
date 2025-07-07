import copy
import numpy as np
import matplotlib.pyplot as plt

xis2_cnt_levels = np.linspace(-20, 20, 101)

outfig = True

prefix = "DDspsp_uni2"
filename = prefix + ".dat"
outname_prefix = prefix

data = np.loadtxt(filename, unpack=True)

sperp = np.unique(data[0])
spara = np.unique(data[1])

nsperp = len(sperp)
nspara = len(spara)

opposite = np.isclose(np.abs(spara[0]), np.abs(spara[-1]))
print("is_opposite : ", opposite)

xi = data[2].reshape((nsperp, nspara))
dd = copy.deepcopy(data[3].reshape((nsperp, nspara)))

s1, s2 = np.meshgrid(sperp, spara, indexing='ij')

ss = s1**2 + s2**2
xis2 = xi * ss

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

"""
s1, s2 = np.meshgrid(sperp, spara, indexing='ij')
ds_perp = np.diff(sperp).mean()
ds_para = np.diff(spara).mean()
bin_volume = 2.0 * np.pi * s1 * ds_perp * ds_para

dd /= bin_volume

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

# dd_cnt_levels = np.linspace(1, dd.max(), 101)

# im = ax.contourf(s1, s2, dd, levels=dd_cnt_levels, cmap="jet")
im = ax.contourf(s1, s2, dd, cmap="jet", levels=101)

plt.colorbar(im, ax=ax, label=r'$DD$')
ax.set_xlabel(r'$s_\perp$ [Mpc/h]', fontsize="xx-large")
ax.set_ylabel(r'$s_\parallel$ [Mpc/h]', fontsize="xx-large")
plt.title(r'$DD$', fontsize="xx-large")
plt.savefig(outname_prefix + "_DD2D_norm.png", bbox_inches="tight")
plt.show()
"""

dd = dd = copy.deepcopy(data[3].reshape((nsperp, nspara)))

s1, s2 = np.meshgrid(sperp, spara, indexing='ij')

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

# im = ax.contourf(s1, s2, dd, levels=dd_cnt_levels, cmap="jet")
im = ax.contourf(s1, s2, dd, cmap="jet", levels=101)

plt.colorbar(im, ax=ax, label=r'$DD$')
ax.set_xlabel(r'$s_\perp$ [Mpc/h]', fontsize="xx-large")
ax.set_ylabel(r'$s_\parallel$ [Mpc/h]', fontsize="xx-large")
plt.title(r'$DD$', fontsize="xx-large")
if outfig: plt.savefig(outname_prefix + "_DD2D.png", bbox_inches="tight")
plt.show()
