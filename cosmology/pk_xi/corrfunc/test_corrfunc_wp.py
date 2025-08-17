import h5py
import numpy as np
import Corrfunc
from Corrfunc.theory import wp

# file_path = "/mnt/work5/stanaka/HaloProptest/pot/l500/n500/halo_props/S003/"
file_path = "/mnt/work5/nbody/grav_pot/GINKAKU/work/l1000/n2000/halo_props/S003/"
file_base = file_path + "halos.0.h5"

data = h5py.File(file_base, "r")
print(data.keys())
print(data["Header"].attrs.keys())
print(data["Halos"].keys())

for i in data["Header"].attrs.keys():
    print(i, " : ", data["Header"].attrs[i])


nfiles = data["Header"].attrs["Nfile"]

x = []
y = []
z = []
mvir = []

for ifile in range(nfiles):
    data = h5py.File(file_path + f"halos.{ifile:d}.h5", "r")

    _x = data["Halos"]["pos"][:, 0]
    _y = data["Halos"]["pos"][:, 1]
    _z = data["Halos"]["pos"][:, 2]
    _m = data["Halos"]["Mvir"]

    x.append(_x)
    y.append(_y)
    z.append(_z)
    mvir.append(_m)

x = np.hstack(x)
y = np.hstack(y)
z = np.hstack(z)
mvir = np.hstack(mvir)

# Setup the problem for wp
boxsize = data["Header"].attrs["BoxSize"]
pimax = 40.0
# nthreads = 4
nthreads = 60


# Setup the bins
rmin = 1.0
rmax = 150.0
nbins = 100

mmin = 1e13
mmax = 1e15

mask = (mvir >= mmin) & (mvir < mmax)
x = x[mask]
y = y[mask]
z = z[mask]

# Create the bins
# rbins = np.logspace(np.log10(0.1), np.log10(rmax), nbins + 1)
rbins = np.linspace(rmin, rmax, nbins + 1)

# Call wp
"""
Corrfunc.theory.wp(boxsize, pimax, nthreads, binfile, X, Y, Z,
weights=None, weight_type=None, verbose=False, output_rpavg=False,
xbin_refine_factor=2, ybin_refine_factor=2, zbin_refine_factor=1,
max_cells_per_dim=100, copy_particles=True, enable_min_sep_opt=True,
c_api_timer=False, c_cell_timer=False, isa='fastest')
"""


pimax = rmax
wp_results = wp(boxsize, pimax, nthreads, rbins, x, y, z, verbose=True, output_ravg=True)

rcen = 0.5 * (wp_results["rmin"] + wp_results["rmax"])
output_file = "corrfunc_wp.dat"

header = "# rcen wp npairs"
np.savetxt(output_file,
           np.column_stack([rcen, wp_results['wp'],
                            wp_results['npairs']]), fmt="%.8e", delimiter=" ", header=header)

print(f"wp results with rcenter saved to {output_file}")
