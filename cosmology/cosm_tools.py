import re
import numpy as np


def atoz(a): return (-1.0 + 1.0 / a)
def ztoa(z): return (1.0 / (1.0 + z))


def Pshot(npart, boxsize):
    # npart : number of 1D particle
    # boxsize : box size [Mpc/h]
    return ((boxsize**3) / (npart**3))


def Nyquist_freq(npart, boxsize):
    # npart : number of 1D particle
    # boxsize : box size [Mpc/h]
    return (np.pi * (1. * npart / boxsize))


def Npart_mass(L=1000, N=1000, Om=0.315):
    H0 = 100.0   # (km/s/Mpc/h)
    G = 4.30091e-9  # (Mpc/h) / (M_sun/h) * (km/s)^2
    rho_mean = (Om * 3.0 * H0**2 / (8.0 * np.pi * G))  # (Msun/h) / (Mpc/h)^3
    V = L**3
    N3 = N * N * N
    m_particle = rho_mean * V / N3  # (Msun/h)
    print(f"Npart={m_particle:.8e} [Msun/h]")
    return m_particle


def read_rockstar_header_from_list(input_file):
    class HaloHeader:
        pass

    header = HaloHeader()
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 30: break
            if line.startswith('#'):
                if 'a =' in line: header.a = float(re.search(r'a = ([0-9.e+-]+)', line).group(1))
                if 'Om =' in line: header.Om = float(re.search(r'Om = ([0-9.e+-]+)', line).group(1))
                if 'Ol =' in line: header.Ol = float(re.search(r'Ol = ([0-9.e+-]+)', line).group(1))
                if 'h =' in line: header.h = float(re.search(r'h = ([0-9.e+-]+)', line).group(1))
                if 'Particle mass:' in line: header.ptcl_mass = float(re.search(r'Particle mass: ([0-9.e+-]+)', line).group(1))
                if 'Box size:' in line: header.boxsize = float(re.search(r'Box size: ([0-9.e+-]+)', line).group(1))
                if 'Force resolution assumed:' in line: header.force_res = float(re.search(r'Force resolution assumed: ([0-9.e+-]+)', line).group(1))

    header.z = atoz(header.a)
    return header


def read_rockstar_mass_from_list(input_file):
    header = read_rockstar_header_from_list(input_file)
    halo_mass = np.loadtxt(input_file, unpack=True)[2]
    return header, halo_mass


def calc_halo_mass_count(mass, log_mmin=10, log_mmax=15, mbin=100):
    bins = np.logspace(log_mmin, log_mmax, mbin + 1)
    count = np.histogram(mass, bins=bins)
    bins_cent = 0.5 * (bins[1:] + bins[:-1])
    return count[0], bins_cent


def calc_hmf_dndm(mass, lbox=1000, log_mmin=10, log_mmax=15, mbin=100):
    bins = np.logspace(log_mmin, log_mmax, mbin + 1)
    count = np.histogram(mass, bins=bins)
    dndm = count[0] / np.diff(count[1]) / (lbox**3)
    bins_cent = 0.5 * (bins[1:] + bins[:-1])
    return dndm, bins_cent


def calc_hmf_dndlnm(mass, lbox=1000, log_mmin=10, log_mmax=15, mbin=100):
    bins = np.logspace(log_mmin, log_mmax, mbin + 1)
    count = np.histogram(mass, bins=bins)
    dndlnm = count[0] / np.diff(np.log(count[1])) / (lbox**3)
    bins_cent = 0.5 * (bins[1:] + bins[:-1])
    return dndlnm, bins_cent


def calc_hmf_dndlog10m(mass, lbox=1000, log_mmin=10, log_mmax=15, mbin=100):
    bins = np.logspace(log_mmin, log_mmax, mbin + 1)
    count = np.histogram(mass, bins=bins)
    dndlog10m = count[0] / np.diff(np.log10(count[1])) / (lbox**3)
    bins_cent = 0.5 * (bins[1:] + bins[:-1])
    return dndlog10m, bins_cent


def analytical_hmf():
    from hmf import MassFunction
    import astropy.units as u

    hmf = MassFunction(cosmo_model="Planck15")
    model = "Tinker08"
    mdef_model = "SOVirial"
    mmin = 7
    mmax = 16
    z = 0.0
    m_nu = [0, 0, 0] * u.eV

    hmf.update(hmf_model=model, mdef_model=mdef_model)
    hmf.update(Mmin=mmin, Mmax=mmax, z=z)
    hmf.cosmo_params = {"m_nu": m_nu}
