import numpy as np

"""
ptcls = np.array([
    [0.1, 0.2, 0.3],  # 0-part:(x,y,z)
    [1.5, 2.2, 3.1],  # 1-part:(x,y,z)
    [4.3, 5.1, 6.4],  # 2-part:(x,y,z)
    ...
])
"""


def ngp(ptcls, nside, lbox):
    mesh = np.zeros((nside, nside, nside))
    grid_spacing = lbox / nside
    positions = ptcls / grid_spacing
    indices = (positions.astype(int) % nside).astype(np.int16)
    np.add.at(mesh, (indices[:, 0], indices[:, 1], indices[:, 2]), 1)
    return mesh


def cic(ptcls, nside, lbox):
    mesh = np.zeros((nside, nside, nside))
    positions = ptcls / lbox * nside - 0.5
    base_indices = (positions + 0.5).astype(np.int16)
    offsets = positions - base_indices

    abs_offsets = np.fabs(offsets)
    sign_offsets = np.sign(offsets).astype(np.int8)
    zeros = np.zeros(len(ptcls)).astype(np.int8)

    for dx, dy, dz in np.ndindex(2, 2, 2):  # 0, 1
        weights = (
            (1.0 - abs_offsets[:, 0] if dx == 0 else abs_offsets[:, 0]) *
            (1.0 - abs_offsets[:, 1] if dy == 0 else abs_offsets[:, 1]) *
            (1.0 - abs_offsets[:, 2] if dz == 0 else abs_offsets[:, 2])
        )

        iidx = np.array([(zeros if dx == 0 else sign_offsets[:, 0]),
                         (zeros if dy == 0 else sign_offsets[:, 1]),
                         (zeros if dz == 0 else sign_offsets[:, 2])]).T
        neighbor_indices = ((base_indices + iidx) % nside).astype(np.int16)
        np.add.at(mesh, (neighbor_indices[:, 0], neighbor_indices[:, 1], neighbor_indices[:, 2]), weights)
    return mesh


def tsc(ptcls, nside, lbox):
    mesh = np.zeros((nside, nside, nside))
    positions = ptcls / lbox * nside - 0.5
    base_indices = (positions + 0.5).astype(np.int16)
    offsets = positions - base_indices

    for dx, dy, dz in np.ndindex(3, 3, 3):  # 0, 1, 2
        dx, dy, dz = np.subtract([dx, dy, dz], 1)   # -1, 0, 1
        weight = (
            (0.5 * (0.5 - offsets[:, 0])**2 if dx == -1 else
             0.75 - (offsets[:, 0])**2 if dx == 0 else
             0.5 * (0.5 + offsets[:, 0])**2
             ) * (
                0.5 * (0.5 - offsets[:, 1])**2 if dy == -1 else
                0.75 - (offsets[:, 1])**2 if dy == 0 else
                0.5 * (0.5 + offsets[:, 1]) ** 2
            ) * (
                0.5 * (0.5 - offsets[:, 2])**2 if dz == -1 else
                0.75 - (offsets[:, 2])**2 if dz == 0 else
                0.5 * (0.5 + offsets[:, 2])**2
            )
        )

        neighbor_indices = (base_indices + np.array([dx, dy, dz])) % nside
        np.add.at(mesh, (neighbor_indices[:, 0], neighbor_indices[:, 1], neighbor_indices[:, 2]), weight)

    return mesh


def main():
    nside = 128
    lbox = 250.0
    npart = nside**3
    ptcls = np.random.uniform(0, lbox, (npart, 3)).astype(np.float32)
    # ptcls = np.random.normal(lbox / 2, 20.0, (npart, 3)).astype(np.float32)

    mesh_p1 = ngp(ptcls, nside, lbox)
    mesh_p2 = cic(ptcls, nside, lbox)
    mesh_p3 = tsc(ptcls, nside, lbox)


if __name__ == "__main__":
    main()
