# Power Spectrum and Correlation Function calculator

## Mass assigments

- Supports
  - NGP (Nearest Grid Point)
  - CIC (Cloud in Cell)
  - TSC (Triangular Shaped Cloud)

- Window correction also supports NGP, CIC and TSC.

## Power spectrum

- auto power spectrum
- cross power spectrum

## Correlation function

- auto correlation function
- cross correlation function

- Supports
  - direct pair count mathod (with cell index sort)
    - LS estimation (Landy and Szalay, 1993, ApJ)
    - Spatially resample blocked jackknife method
    - Randomly resampled blocked jackknife method

  - inverse FFT method (with cell index sort)
    - Spatially resample blocked jackknife method

## Running Sample

### halo auto correlation function

```sh
./halo_xi -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 150  -o output.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 150  --est RR --nrand_factor 2 -o output.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 150  --est LS --nrand_factor 2 -o output.dat
```

- default estimator is `ideal` : $\xi(r)=DD(r)/\mathrm{ideal pair count} - 1$
- `RR` (Peebles–Hauser) estimator : $\xi(r)=DD(r)/RR(r) - 1$
- `LS` estimator : $\xi(r)=(DD(r) -2DR(r) + RR(r))/RR(r)$
