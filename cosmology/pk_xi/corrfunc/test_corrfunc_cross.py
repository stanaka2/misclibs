import h5py
import numpy as np
import Corrfunc
from Corrfunc.theory import xi, DD

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

mmin1, mmax1 = 1e13, 1e14
mmin2, mmax2 = 1e14, 1e15

mask1 = (mvir >= mmin1) & (mvir < mmax1)
x1 = x[mask1]
y1 = y[mask1]
z1 = z[mask1]

mask2 = (mvir >= mmin2) & (mvir < mmax2)
x2 = x[mask2]
y2 = y[mask2]
z2 = z[mask2]

rand_N1 = len(x1)
rand_x1 = np.random.uniform(0, boxsize, rand_N1).astype(np.float32)
rand_y1 = np.random.uniform(0, boxsize, rand_N1).astype(np.float32)
rand_z1 = np.random.uniform(0, boxsize, rand_N1).astype(np.float32)

rand_N2 = len(x2)
rand_x2 = np.random.uniform(0, boxsize, rand_N2).astype(np.float32)
rand_y2 = np.random.uniform(0, boxsize, rand_N2).astype(np.float32)
rand_z2 = np.random.uniform(0, boxsize, rand_N2).astype(np.float32)

# Create the bins
# rbins = np.logspace(np.log10(0.1), np.log10(rmax), nbins + 1)
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

D1D2 = DD(autocorr=False, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
          X1=x1, Y1=y1, Z1=z1, X2=x2, Y2=y2, Z2=z2,
          verbose=True, output_ravg=True)

R1R2 = DD(autocorr=False, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
          X1=rand_x1, Y1=rand_y1, Z1=rand_z1, X2=rand_x2, Y2=rand_y2, Z2=rand_z2,
          verbose=True, output_ravg=True)

D1R2 = DD(autocorr=False, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
          X1=x1, Y1=y1, Z1=z1, X2=rand_x2, Y2=rand_y2, Z2=rand_z2,
          verbose=True, output_ravg=True)

D2R1 = DD(autocorr=False, boxsize=boxsize, nthreads=nthreads, binfile=rbins,
          X1=x2, Y1=y2, Z1=z2, X2=rand_x1, Y2=rand_y1, Z2=rand_z1,
          verbose=True, output_ravg=True)

ND1, ND2 = len(x1), len(x2)
NR1, NR2 = len(rand_x1), len(rand_x2)

DD12 = D1D2['npairs'].astype(float)
D1R2 = D1R2['npairs'].astype(float)
D2R1 = D2R1['npairs'].astype(float)
RR12 = R1R2['npairs'].astype(float)

DDhat = DD12 / (ND1 * ND2)
D1R2hat = D1R2 / (ND1 * NR2)
D2R1hat = D2R1 / (ND2 * NR1)
RRhat = RR12 / (NR1 * NR2)

xi_cross = (DDhat - D1R2hat - D2R1hat + RRhat) / RRhat

rcen = 0.5 * (D1D2["rmin"] + D1D2["rmax"])
output_file = "corrfunc_xi_cross.dat"
header = "# rcen xi npairs"
np.savetxt(output_file,
           np.column_stack([rcen, xi_cross]), fmt="%.8e", delimiter=" ", header=header)

print(f"xi results with rcenter saved to {output_file}")
