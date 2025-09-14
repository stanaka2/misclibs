import h5py
import numpy as np
import Corrfunc
from Corrfunc.theory import xi, DD

nfactor = 1

np.random.seed(12345)

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
nbins = 21

mmin, mmax = 1e13, 1e20

mask1 = (mvir >= mmin) & (mvir < mmax)
x1 = x[mask1]
y1 = y[mask1]
z1 = z[mask1]

rand_N1 = len(x1) * nfactor
rand_x1 = np.random.uniform(0, boxsize, rand_N1).astype(np.float32)
rand_y1 = np.random.uniform(0, boxsize, rand_N1).astype(np.float32)
rand_z1 = np.random.uniform(0, boxsize, rand_N1).astype(np.float32)

# Create the bins
# rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
rbins = np.linspace(rmin, rmax, nbins + 1)

# Call xi
"""
def xi(boxsize, nthreads, binfile, X, Y, Z,
       weights=None, weight_type=None, verbose=False, output_ravg=False,
       xbin_refine_factor=2, ybin_refine_factor=2,
       zbin_refine_factor=1, max_cells_per_dim=100,
       copy_particles=True, enable_min_sep_opt=True,
       c_api_timer=False, isa=r'fastest'):

Corrfunc.theory.DD(autocorr, nthreads, binfile, X1, Y1, Z1,
weights1=None, periodic=True, boxsize=None, X2=None, Y2=None, Z2=None,
weights2=None, verbose=False, output_ravg=False,
xbin_refine_factor=2, ybin_refine_factor=2, zbin_refine_factor=1,
max_cells_per_dim=100, copy_particles=True, enable_min_sep_opt=True,
c_api_timer=False, isa='fastest', weight_type=None)


"""

DDp = DD(autocorr=True, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
         X1=x1, Y1=y1, Z1=z1,
         verbose=True, output_ravg=True)

RRp = DD(autocorr=True, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
         X1=rand_x1, Y1=rand_y1, Z1=rand_z1,
         verbose=True, output_ravg=True)

DRp = DD(autocorr=False, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
         X1=x1, Y1=y1, Z1=z1, X2=rand_x1, Y2=rand_y1, Z2=rand_z1,
         verbose=True, output_ravg=True)


ND = len(x1)
NR = len(rand_x1)

DDpp = DDp['npairs'].astype(float)
RRpp = RRp['npairs'].astype(float)
DRpp = DRp['npairs'].astype(float)

DDhat = DDpp / (ND * ND)
DRhat = DRpp / (ND * NR)
RRhat = RRpp / (NR * NR)

xi_cross = (DDhat - DRhat - DRhat + RRhat) / RRhat

rcen = 0.5 * (DDp["rmin"] + DDp["rmax"])
output_file = f"corrfunc_xi_cross_LS_n{nfactor}.dat"
header = "# rcen xi DD DR RR"
np.savetxt(output_file,
           np.column_stack([rcen, xi_cross, DDpp, DRpp, RRpp]), fmt="%.8e", delimiter=" ", header=header)

print(f"xi results with rcenter saved to {output_file}")
