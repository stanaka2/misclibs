import ctypes
import pathlib
import numpy as np
from tqdm import tqdm

import Corrfunc
from Corrfunc.theory import xi

# file_path = "/mnt/work5/stanaka/HaloProptest/pot/l500/n500/halo_props/S003/"
# file_path = "/mnt/work5/nbody/grav_pot/GINKAKU/work/l1000/n2000/halo_props/S003/"
# file_base = file_path + "halos.0.h5"

file_prefix = "/mnt/work5/stanaka/HaloProptest/run/snapdir_010/snapshot_010"


class GadgetHeader(ctypes.Structure):
    npart_local = 0
    npart_total = 0
    hsize = 256
    hunuse = hsize - 14 * ctypes.sizeof(ctypes.c_double) - 11 * ctypes.sizeof(ctypes.c_int32) - 12 * ctypes.sizeof(ctypes.c_uint32)
    _pack_ = 1
    _fields_ = (
        ('npart', ctypes.c_uint32 * 6),
        ('mass', ctypes.c_double * 6),
        ('anow', ctypes.c_double),
        ('znow', ctypes.c_double),
        ('flag_sfr', ctypes.c_int32),
        ('flag_feed', ctypes.c_int32),
        ('npartTotal', ctypes.c_uint32 * 6),
        ('flag_cooling', ctypes.c_int32),
        ('num_files', ctypes.c_int32),
        ('box_size', ctypes.c_double),
        ('omega_m', ctypes.c_double),
        ('omega_lambda', ctypes.c_double),
        ('h0', ctypes.c_double),
        ('unused_d', ctypes.c_double * 2),
        ('unused_i', ctypes.c_int32 * 4),
        ('step', ctypes.c_int32),
        ('output_indx', ctypes.c_int32),
        ('output_pot', ctypes.c_int32),
        ('unused', ctypes.c_char * hunuse),
    )


class GadgetData:
    def __init__(self):
        self.unset_load_all()

    def set_load_all(self):
        self.load_vel = True
        self.load_index = True
        self.load_pot = True
        self.load_dens = True
        self.load_pot_grad = True
        self.load_dens_grad = True

    def unset_load_all(self):
        self.load_vel = False
        self.load_index = False
        self.load_pot = False
        self.load_dens = False
        self.load_pot_grad = False
        self.load_dens_grad = False

    def read_header(self, input_prefix):
        gheader = GadgetHeader()
        gheader.input_prefix = input_prefix

        if pathlib.Path(input_prefix + ".0").is_file():
            input_base = gheader.input_prefix + ".0"
            self.hir_dir = False
        elif pathlib.Path(input_prefix + ".0/0").is_file():
            input_base = gheader.input_prefix + ".0/0"
            self.hir_dir = True
        else:
            assert False, "Not found input file"

        dummy_size = ctypes.sizeof(ctypes.c_int32)

        read_file = open(input_base, "rb")
        read_file.read(dummy_size)
        read_file.readinto(gheader)
        read_file.read(dummy_size)
        read_file.close()

        # if gadget_type == "GADGET":
        #    gheader.npart_total = (gheader.npartTotal_hw[1] << 32) + gheader.npartTotal[1]
        # elif gadget_type == "LGADGET":
        gheader.npart_total = (gheader.npartTotal[2] << 32) + gheader.npartTotal[1]
        self.gheader = gheader

    def read_gdt_data(self, pos_range=None):
        """
        # Mpc/h scale
        range = ([0,1000],[0,1000],[0,10])
        """
        if pos_range == None:
            xrange = -1e+8, 1e+8
            yrange = -1e+8, 1e+8
            zrange = -1e+8, 1e+8
        else:
            xrange = pos_range[0]
            yrange = pos_range[1]
            zrange = pos_range[2]

        dummy_size = ctypes.sizeof(ctypes.c_int32)
        tmp_header = GadgetHeader()
        self.gheader.npart_local = 0

        f32_type = np.dtype(np.float32)
        f32_vec_type = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
        i64_type = np.dtype([("index", np.int64)])

        pos, vel = [], []
        index = []

        for ifile in tqdm(range(self.gheader.num_files)):
            if self.hir_dir:
                filename = "{0:s}.{1:d}/{2:d}".format(self.gheader.input_prefix, ifile // 1000, ifile)
            else:
                filename = "{0:s}.{1:d}".format(self.gheader.input_prefix, ifile)

            read_file = open(filename, "rb")

            read_file.read(dummy_size)
            read_file.readinto(tmp_header)
            read_file.read(dummy_size)

            npart_loc = tmp_header.npart[1]

            read_file.read(dummy_size)
            _pos = np.fromfile(read_file, dtype=f32_vec_type, count=int(npart_loc))
            read_file.read(dummy_size)

            if self.load_vel:
                read_file.read(dummy_size)
                _vel = np.fromfile(read_file, dtype=f32_vec_type, count=int(npart_loc))
                read_file.read(dummy_size)
            else:
                skip_size = 2 * dummy_size + int(npart_loc) * f32_vec_type.itemsize
                read_file.seek(skip_size, 1)

            if self.load_index:
                read_file.read(dummy_size)
                _index = np.fromfile(read_file, dtype=i64_type, count=int(npart_loc))
                read_file.read(dummy_size)
            else:
                skip_size = 2 * dummy_size + int(npart_loc) * i64_type.itemsize
                read_file.seek(skip_size, 1)

            read_file.close()

            pos_idx = np.where(
                ((xrange[0] <= _pos["x"]) & (_pos["x"] <= xrange[1])) &
                ((yrange[0] <= _pos["y"]) & (_pos["y"] <= yrange[1])) &
                ((zrange[0] <= _pos["z"]) & (_pos["z"] <= zrange[1])))[0]

            _pos = _pos[pos_idx]
            pos.append(_pos)

            if self.load_vel:
                _vel = _vel[pos_idx]
                vel.append(_vel)

            if self.load_index:
                _index = _index[pos_idx]
                index.append(_index)

            npart_loc = len(pos_idx)
            self.gheader.npart_local += npart_loc

        self.pos = np.hstack(pos)
        del pos

        if self.load_vel:
            self.vel = np.hstack(vel)
            del vel
        if self.load_index:
            self.index = np.hstack(index)
            del index


data = GadgetData()
data.read_header(file_prefix)
data.read_gdt_data()

boxsize = data.gheader.box_size
pimax = 20.0
nthreads = 32

# Setup the bins
rmin = 0.0
rmax = 20.0
nbins = 100

x = data.pos["x"]
y = data.pos["y"]
z = data.pos["z"]

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
"""

xi_results = xi(boxsize, nthreads, rbins, x, y, z, verbose=True, output_ravg=True)

rcen = 0.5 * (xi_results["rmin"] + xi_results["rmax"])
output_file = "corrfunc_mma_xi.dat"

header = "# rcen xi npairs"
np.savetxt(output_file,
           np.column_stack([rcen, xi_results['xi'],
                            xi_results['npairs']]), fmt="%.8e", delimiter=" ", header=header)

print(f"xi results with rcenter saved to {output_file}")
