import struct
import numpy as np
import matplotlib.pyplot as plt


def read_density(filename):
    with open(filename, 'rb') as f:
        header_fmt = '<11i'
        header_bytes = struct.calcsize(header_fmt)
        hdr = struct.unpack(header_fmt, f.read(header_bytes))
        (ntasks, ntasks_x, ntasks_y, ntasks_z,
         thistask, thistask_x, thistask_y, thistask_z,
         nx, ny, nz) = hdr

        count = nx * ny * nz
        dens = np.fromfile(f, dtype='<f4', count=count)
    dens = dens.reshape((nx, ny, nz))

    header = {
        'ntasks': ntasks, 'ntasks_x': ntasks_x, 'ntasks_y': ntasks_y, 'ntasks_z': ntasks_z,
        'thistask': thistask, 'thistask_x': thistask_x, 'thistask_y': thistask_y, 'thistask_z': thistask_z,
        'nx': nx, 'ny': ny, 'nz': nz
    }
    return dens, header


if __name__ == "__main__":
    fname = "no_mpi_dens_0"
    rho, meta = read_density(fname)

    fig = plt.figure(figsize=(6, 6))

    zc = meta['nz'] // 2
    plt.imshow(rho[:, :, zc].T, origin='lower', aspect='equal')
    plt.colorbar(label='density')
    plt.title(f"{fname}  z-slice {zc}")
    plt.xlabel('ix'); plt.ylabel('iy')
    plt.tight_layout()
    plt.show()
