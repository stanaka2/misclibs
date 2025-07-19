import sys
import numpy as np
import matplotlib.pyplot as plt


def read_field(filename):
    with open(filename, 'rb') as f:
        nmesh = np.fromfile(f, dtype=np.int32, count=1)[0]
        nmesh_tot = np.fromfile(f, dtype=np.int64, count=1)[0]
        lbox = np.fromfile(f, dtype=np.float64, count=1)[0]
        mesh = np.fromfile(f, dtype=np.float32, count=nmesh_tot)

    mesh.resize(nmesh, nmesh, nmesh)
    return nmesh, nmesh_tot, lbox, mesh


filename = sys.argv[1]
plot_type = sys.argv[2]  # dens, pot

nmesh, nmesh_tot, lbox, mesh = read_field(filename)
print(f'nmesh: {nmesh}, nmesh_tot: {nmesh_tot}, lbox: {lbox}')

nslice = nmesh // 2
extent = [0, lbox, 0, lbox]

if plot_type == "dens":
    slice_mesh = np.log10(mesh[:, :, nslice] + 1.0)
    clab = r"$\log_{10} \rho/\bar{\rho}$"
    vmin, vmax = -2, 3
elif plot_type == "pot":
    slice_mesh = -mesh[:, :, nslice] * 1e+5
    clab = r"$-\phi \times 10^5$"
    vmin, vmax = None, None

# Plot the slice
plt.figure(figsize=(8, 8))
plt.imshow(slice_mesh, vmin=vmin, vmax=vmax, extent=extent, origin='lower', interpolation="none", cmap='jet')
plt.colorbar(label=clab)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('XY Plane Slice')
plt.show()
