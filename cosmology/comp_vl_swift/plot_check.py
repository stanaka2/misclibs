import ctypes
import struct
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import brentq
import concurrent.futures as confu

pc = 3.0856775807e+18
mpc = pc * 1.0e+6


def dtda(a, cosm):
    om, ov = cosm.omega_m, cosm.omega_v
    ok = 1.0 - om - ov
    return np.sqrt(a / (om + ok * a + ov * a**3))


def atotime(a, cosm):
    result, _ = quad(dtda, 0, a, args=(cosm,))
    return result


def timetoa(t, cosm):
    def func(a): return atotime(a, cosm) - t
    return brentq(func, 1e-4, 2.0)


def timetoz(t, cosm):
    a = timetoa(t, cosm)
    return 1.0 / a - 1.0


class ReprStructure(ctypes.Structure):
    def __repr__(self) -> str:
        values = ", ".join(f"{name}={value}" for name, value in self._asdict().items())
        return f"<{self.__class__.__name__}: {values}>"

    def _asdict(self) -> dict:
        return {field[0]: getattr(self, field[0]) for field in self._fields_}


class Meshes(ReprStructure):
    _pack_ = 1
    _fields_ = (
        ('x_total', ctypes.c_int32), ('y_total', ctypes.c_int32), ('z_total', ctypes.c_int32),
        ('x_local', ctypes.c_int32), ('y_local', ctypes.c_int32), ('z_local', ctypes.c_int32),
        ('x_loc_start', ctypes.c_int32), ('x_loc_end', ctypes.c_int32),
        ('y_loc_start', ctypes.c_int32), ('y_loc_end', ctypes.c_int32),
        ('z_loc_start', ctypes.c_int32), ('z_loc_end', ctypes.c_int32),
        ('loc_size', ctypes.c_int64),
    )


class NeutrinoParam(ReprStructure):
    # _pack_ = 1
    _fields_ = (
        ('mass_nu_mun', ctypes.c_int32),
        ('deg', ctypes.c_int * 3), ('sum_mass', ctypes.c_float),
        ('mass', ctypes.c_float * 3), ('frac', ctypes.c_float * 3)
    )


class CosmoParam(ReprStructure):
    # _pack_ = 1
    _fields_ = (
        ('omega_m', ctypes.c_float), ('omega_v', ctypes.c_float), ('omega_b', ctypes.c_float),
        ('omega_nu', ctypes.c_float), ('omega_r', ctypes.c_float),
        ('hubble', ctypes.c_float), ('tend', ctypes.c_float),
        ('nu', NeutrinoParam)
    )


class BaseHeader(ReprStructure):
    _pack_ = 1
    _fields_ = (
        ### base header ###
        ('cosmology_flag', ctypes.c_int32),
        ('canonical_flag', ctypes.c_int32),

        ('nnode_x', ctypes.c_int32), ('nnode_y', ctypes.c_int32), ('nnode_z', ctypes.c_int32),
        ('mpi_proc', ctypes.c_int32), ('mpi_rank', ctypes.c_int32),
        ('rank_x', ctypes.c_int32), ('rank_y', ctypes.c_int32), ('rank_z', ctypes.c_int32),

        ('xmin', ctypes.c_float), ('xmax', ctypes.c_float),
        ('ymin', ctypes.c_float), ('ymax', ctypes.c_float),
        ('zmin', ctypes.c_float), ('zmax', ctypes.c_float),
        ('xmin_local', ctypes.c_float), ('xmax_local', ctypes.c_float),
        ('ymin_local', ctypes.c_float), ('ymax_local', ctypes.c_float),
        ('zmin_local', ctypes.c_float), ('zmax_local', ctypes.c_float),

        ('lunit', ctypes.c_double), ('munit', ctypes.c_double), ('tunit', ctypes.c_double),
        ('tnow', ctypes.c_float), ('dtime', ctypes.c_float), ('step', ctypes.c_int32),
        ### base header ###
    )


class RunParamCOSM(ReprStructure):
    _pack_ = 1
    _fields_ = BaseHeader._fields_ + (
        ('cosm', CosmoParam),
        ('nm', Meshes),
    )


class PtclParamCOSM(ReprStructure):
    _pack_ = 1
    _fields_ = BaseHeader._fields_ + (
        ('cosm', CosmoParam),
        ('npart', ctypes.c_uint64),
        ('npart_total', ctypes.c_uint64),
    )


def unpack_i16(i16):
    return struct.unpack('<h', i16)[0]


def unpack_i32(i32):
    return struct.unpack('<i', i32)[0]


def unpack_i64(i64):
    return struct.unpack('<q', i64)[0]


def unpack_f16(f16):
    return struct.unpack('<e', f16)[0]


def unpack_f32(f32):
    return struct.unpack('<f', f32)[0]


def unpack_f64(f64):
    return struct.unpack('<d', f64)[0]


class ReadData:
    def check_flag(self, input_file):
        read_file = open(input_file, "rb")
        cosmology_flag = read_file.read(ctypes.sizeof(ctypes.c_int32))
        cosmology_flag = struct.unpack('<i', cosmology_flag)[0]
        canonical_flag = read_file.read(ctypes.sizeof(ctypes.c_int32))
        canonical_flag = struct.unpack('<i', canonical_flag)[0]
        read_file.close()

        self.cosmology_flag = cosmology_flag
        self.canonical_flag = canonical_flag

    def read_header(self, input_file):
        self.check_flag(input_file)
        this_run = RunParamCOSM()
        head_size = ctypes.sizeof(RunParamCOSM)

        read_file = open(input_file, "rb")

        read_file.readinto(this_run)
        read_file.close()

        this_run.delta_x = (this_run.xmax - this_run.xmin) / this_run.nm.x_total
        this_run.delta_y = (this_run.ymax - this_run.ymin) / this_run.nm.y_total
        this_run.delta_z = (this_run.zmax - this_run.zmin) / this_run.nm.z_total

        if this_run.cosmology_flag:
            this_run.znow = timetoz(this_run.tnow, this_run.cosm)
            this_run.anow = timetoa(this_run.tnow, this_run.cosm)
            this_run.box_size = this_run.lunit * this_run.cosm.hubble / mpc
            this_run.vunit = 1.0

        else:
            this_run.box_size = 1
            this_run.lunit = 1.0
            this_run.tunit = 1.0
            this_run.munit = 1.0
            this_run.vunit = this_run.lunit / this_run.tunit

        return this_run, head_size

    def read_single_volume(self, input_file):
        base_run, head_size = self.read_header(input_file)
        volume = np.zeros((base_run.nm.x_total, base_run.nm.y_total, base_run.nm.z_total), dtype=np.float32)

        tmp_run, head_size = self.read_header(input_file)
        loc_num = tmp_run.nm.x_local * tmp_run.nm.y_local * tmp_run.nm.z_local

        read_file = open(input_file, "rb")
        read_file.read(head_size)

        # print(input_file)
        loc_vol = np.fromfile(read_file, dtype=np.float32, count=int(loc_num))
        read_file.close()

        loc_vol.resize(tmp_run.nm.x_local, tmp_run.nm.y_local, tmp_run.nm.z_local)

        volume[tmp_run.nm.x_loc_start:tmp_run.nm.x_loc_end,
               tmp_run.nm.y_loc_start:tmp_run.nm.y_loc_end,
               tmp_run.nm.z_loc_start:tmp_run.nm.z_loc_end] = loc_vol

        self.volume = volume
        return base_run

    def read_multiple_volume(self, input_file, n):
        base_run, head_size = self.read_header(input_file)

        volumes = [0] * n
        for ii in range(n):
            volumes[ii] = np.zeros((base_run.nm.x_total, base_run.nm.y_total, base_run.nm.z_total), dtype=np.float32)

        tmp_run, head_size = self.read_header(input_file)
        loc_num = tmp_run.nm.x_local * tmp_run.nm.y_local * tmp_run.nm.z_local
        loc_vols = [0] * n

        read_file = open(input_file, "rb")
        read_file.read(head_size)

        for ii in range(n):
            loc_vols[ii] = np.fromfile(read_file, dtype=np.float32, count=int(loc_num))
        read_file.close()

        for ii in range(n):
            loc_vols[ii].resize(tmp_run.nm.x_local, tmp_run.nm.y_local, tmp_run.nm.z_local)

        for ii in range(n):
            volumes[ii][tmp_run.nm.x_loc_start:tmp_run.nm.x_loc_end,
                        tmp_run.nm.y_loc_start:tmp_run.nm.y_loc_end,
                        tmp_run.nm.z_loc_start:tmp_run.nm.z_loc_end] = loc_vols[ii]

        self.volumes = volumes
        return base_run


def calc_volume_sigma(ix, sxx, sxy, sxz, syy, syz, szz):
    y, z = sxx.shape[1], sxx.shape[2]
    S = np.empty((y, z, 3, 3), dtype=sxx.dtype)

    S[..., 0, 0] = sxx[ix]
    S[..., 0, 1] = sxy[ix]; S[..., 1, 0] = sxy[ix]
    S[..., 0, 2] = sxz[ix]; S[..., 2, 0] = sxz[ix]
    S[..., 1, 1] = syy[ix]
    S[..., 1, 2] = syz[ix]; S[..., 2, 1] = syz[ix]
    S[..., 2, 2] = szz[ix]

    w = np.linalg.eigvalsh(S)
    s11_ix = w[..., 0]
    s22_ix = w[..., 1]
    s33_ix = w[..., 2]
    return ix, s11_ix, s22_ix, s33_ix


def set_volume_sigma(input_file, nproc=4):
    rd = ReadData()
    this_run = rd.read_multiple_volume(input_file, 6)
    sxx, sxy, sxz, syy, syz, szz = rd.volumes

    sxx = np.asarray(sxx); sxy = np.asarray(sxy); sxz = np.asarray(sxz)
    syy = np.asarray(syy); syz = np.asarray(syz); szz = np.asarray(szz)

    lenx, leny, lenz = sxx.shape
    s11 = np.empty((lenx, leny, lenz), dtype=sxx.dtype)
    s22 = np.empty_like(s11)
    s33 = np.empty_like(s11)

    with confu.ProcessPoolExecutor(max_workers=nproc) as executor:
        futures = [
            executor.submit(calc_volume_sigma, ix, sxx, sxy, sxz, syy, syz, szz)
            for ix in range(lenx)
        ]
        for fut in confu.as_completed(futures):
            ix, s11_ix, s22_ix, s33_ix = fut.result()
            s11[ix] = s11_ix
            s22[ix] = s22_ix
            s33[ix] = s33_ix

    sigma = np.sqrt(s11 + s22 + s33)
    sigma /= np.mean(sigma)
    return this_run, sigma


def roll_center(slice_data, target_x=75, target_y=190):
    center_x = slice_data.shape[1] // 2
    center_y = slice_data.shape[0] // 2
    shift_x = center_x - target_x
    shift_y = center_y - target_y
    slice_data = np.roll(slice_data, shift=(shift_y, shift_x), axis=(0, 1))
    return slice_data


dens_cmap = "jet"
velc_cmap = "seismic"
sigma_cmap = "magma"


if __name__ == "__main__":

    dens_file = "z0/vlasov_dens_nb1"
    dens_file = "./swift_dens_nb1"
    dens_vmin, dens_vmax = -0.1, 0.4

    velc_file = "z0/vlasov_velc_nb1"
    velc_vmin, velc_vmax = -500, 500

    sigma_file = "z0/vlasov_sigma_nb1"
    sigma_vmin, sigma_vmax = 0.8, 1.2

    z_center = 192 // 2

    if 1:
        rd = ReadData()
        this_run = rd.read_single_volume(dens_file)
        dens = rd.volume
        dens = dens / np.mean(dens)

        slice_data = dens[:, :, z_center]
        slice_data = roll_center(slice_data)

        plt.figure(figsize=(6, 5))
        plt.imshow(np.log10(slice_data + 1e-5), origin='lower', interpolation="none", cmap=dens_cmap,
                   vmin=dens_vmin, vmax=dens_vmax)
        plt.colorbar(label='density log10(rho/rho_mean)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    if 0:
        rd = ReadData()
        this_run = rd.read_multiple_volume(velc_file, 3)
        velc = rd.volumes

        slice_data = velc[0][:, :, z_center]
        slice_data = roll_center(slice_data)

        plt.figure(figsize=(6, 5))
        plt.imshow(slice_data, origin='lower', interpolation="none", cmap=velc_cmap,
                   vmin=velc_vmin, vmax=velc_vmax)
        plt.colorbar(label='velocity vx [km/s]')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    if 1:
        this_run, sigma_norm = set_volume_sigma(sigma_file, 12)

        slice_data = sigma_norm[:, :, z_center]
        slice_data = roll_center(slice_data)

        plt.figure(figsize=(6, 5))
        plt.imshow(slice_data, origin='lower', interpolation="none", cmap=sigma_cmap, vmin=sigma_vmin, vmax=sigma_vmax)
        plt.colorbar(label='velocity dispersion sigma^2 / sigma_mean^2')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
