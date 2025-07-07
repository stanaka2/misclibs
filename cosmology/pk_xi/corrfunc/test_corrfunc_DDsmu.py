import h5py
import numpy as np
import Corrfunc
from Corrfunc.theory import DDsmu

# file_path = "/mnt/work5/stanaka/HaloProptest/pot/l500/n500/halo_props/S003/"
file_path = "/mnt/work5/nbody/grav_pot/GINKAKU/work/l1000/n2000/halo_props/S003/"
file_prefix = "halos"

file_path = "/mnt/work5/stanaka/test/uniform/halos/"
file_prefix = "halos_uniform"

file_base = file_path + file_prefix + ".0.h5"

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
    data = h5py.File(file_path + f"{file_prefix}.{ifile:d}.h5", "r")

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
# rmin = 1.0
rmin = 1.0e-4
rmax = 150.0
nbins = 100

mumax = 1.0
mu_nbins = 100

mmin = 1e13
mmax = 1e15

mask = (mvir >= mmin) & (mvir < mmax)
x = x[mask]
y = y[mask]
z = z[mask]

ngrp = len(x)

# for RR data

rng = np.random.default_rng(seed=100)
nrand_factor = 2
nrand = nrand_factor * ngrp

rx = rng.uniform(0.0, boxsize, nrand).astype(np.float32)
ry = rng.uniform(0.0, boxsize, nrand).astype(np.float32)
rz = rng.uniform(0.0, boxsize, nrand).astype(np.float32)


# Create the bins
# rbins = np.logspace(np.log10(0.1), np.log10(rmax), nbins + 1)
rbins = np.linspace(rmin, rmax, nbins + 1)
mubins = np.linspace(0, mumax, mu_nbins + 1)

# Call xi(s,mu)
"""
Corrfunc.theory.DDsmu(autocorr, nthreads, binfile, mu_max, nmu_bins, X1, Y1, Z1,
                        weights1=None, periodic=True, boxsize=None, X2=None, Y2=None, Z2=None,
                        weights2=None, verbose=False, output_savg=False, fast_divide_and_NR_steps=0,
                        xbin_refine_factor=2, ybin_refine_factor=2, zbin_refine_factor=1, max_cells_per_dim=100,
                        copy_particles=True, enable_min_sep_opt=True, c_api_timer=False, isa='fastest', weight_type=None)
"""

DDsmu_results = DDsmu(autocorr=True, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
                      mu_max=mumax, nmu_bins=mu_nbins, X1=x, Y1=y, Z1=z, verbose=True)

RRsmu_results = DDsmu(autocorr=True, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
                      mu_max=mumax, nmu_bins=mu_nbins, X1=rx, Y1=ry, Z1=rz, verbose=True)

xi2D = (nrand_factor**2 * DDsmu_results['npairs']) / RRsmu_results['npairs'] - 1.0

rcen_tile = 0.5 * (DDsmu_results["smin"] + DDsmu_results["smax"])
mucen = 0.5 * (mubins[:-1] + mubins[1:])
mucen_tile = np.tile(mucen, nbins)

output_file = "corrfunc_DDsmu.dat"
header = "# rcen mucen xi npairs"

np.savetxt(output_file,
           np.column_stack([rcen_tile, mucen_tile, xi2D, DDsmu_results['npairs']]),
           fmt=("%.6e", "%.6e", "%.10e", "%.10e"), delimiter=" ", header=header)

print(f"DD(s,mu) results with rcenter saved to {output_file}")
