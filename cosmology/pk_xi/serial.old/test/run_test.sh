#!/usr/bin/bash

halo_prefix=/mnt/work5/nbody/grav_pot/GINKAKU/work/l1000/n1000/halo_props/S003/halos
ptcl_prefix=/mnt/work5/nbody/grav_pot/GINKAKU/work/l1000/n1000/snapdir_potdens_pm4.00_cr12.0_theta0.0_sft0_003/snapshot_003

OUTPUTDIR=test_output
mkdir -p ${OUTPUTDIR}


echo "halo_xi : test"

if true ; then
echo "halo_xi : basic r-bin tests"
./halo_xi -i ${halo_prefix} --mrange 13 15 -o ${OUTPUTDIR}/halo_xi_00.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 50 -o ${OUTPUTDIR}/halo_xi_01.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0.01 120 --log_bin -o ${OUTPUTDIR}/halo_xi_02.dat
fi

if true ; then
echo "halo_xi : estimator method tests"
./halo_xi -i ${halo_prefix} --mrange 13 15 --est ideal -o ${OUTPUTDIR}/halo_xi_10.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --est RR --nrand_factor 2 -o ${OUTPUTDIR}/halo_xi_11.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --est LS --nrand_factor 1 -o ${OUTPUTDIR}/halo_xi_12.dat
fi

if false ; then
echo "halo_xi : RSD / Gred tests"
./halo_xi -i ${halo_prefix} --mrange 13 15 -o ${OUTPUTDIR}/halo_xi_20.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --RSD -o ${OUTPUTDIR}/halo_xi_21.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --Gred -o ${OUTPUTDIR}/halo_xi_22.dat
./halo_xi -i ${halo_prefix} --mrange 13 15 --RSD --Gred -o ${OUTPUTDIR}/halo_xi_23.dat
fi

echo "halo_xi2D : test"
echo "Support (s,mu) binning and (s_perp,s_para) binning"
echo "Not support log-bin"

if true ; then
echo "halo_xi2D : basic r-bin tests"
./halo_xi2D -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 50 --mode smu --est LS  --nrand_factor 2 --half_angle -o ${OUTPUTDIR}/halo_xi2D_smu0.dat
./halo_xi2D -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 50 --mode smu --est LS  --nrand_factor 2 -o ${OUTPUTDIR}/halo_xi2D_smu1.dat
./halo_xi2D -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 50 --mode spsp --est LS --nrand_factor 2 --half_angle -o ${OUTPUTDIR}/halo_xi2D_spsp0.dat
./halo_xi2D -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 50 --mode spsp --est LS --nrand_factor 2 -o ${OUTPUTDIR}/halo_xi2D_spsp1.dat
./halo_xi2D -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 50 --mode smu --est LS --nrand_factor 2 --RSD --Gred --half_angle -o ${OUTPUTDIR}/halo_xi2D_smu_shift.dat
./halo_xi2D -i ${halo_prefix} --mrange 13 15 --nr 100 --rrange 0 50 --mode spsp --est LS --nrand_factor 2 --RSD --Gred --half_angle -o ${OUTPUTDIR}/halo_xi2D_spsp_shift.dat
fi

echo "matter_xi : test"

if true ; then
echo "matter_xi : basic r-bin tests"
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 -o ${OUTPUTDIR}/matter_xi_00.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 --nr 100 --rrange 0 50 -o ${OUTPUTDIR}/matter_xi_01.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 --nr 100 --rrange 0.01 120 --log_bin -o ${OUTPUTDIR}/matter_xi_02.dat
fi

if true ; then
echo "matter_xi : sampling rate tests"
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.0005 -o ${OUTPUTDIR}/matter_xi_10.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 -o ${OUTPUTDIR}/matter_xi_11.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.002 -o ${OUTPUTDIR}/matter_xi_12.dat
fi


if true ; then
echo "matter_xi : estimator method tests"
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 --est ideal -o ${OUTPUTDIR}/matter_xi_20.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 --est RR --nrand_factor 2 -o ${OUTPUTDIR}/matter_xi_21.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 --est LS --nrand_factor 1 -o ${OUTPUTDIR}/matter_xi_22.dat
fi

if false ; then
echo "matter_xi : RSD / Gred tests"
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 -o ${OUTPUTDIR}/matter_xi_30.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 --RSD -o ${OUTPUTDIR}/matter_xi_31.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 --Gred -o ${OUTPUTDIR}/matter_xi_32.dat
./matter_xi -i ${ptcl_prefix} --sampling_rate 0.001 --RSD --Gred -o ${OUTPUTDIR}/matter_xi_33.dat
fi

echo "matter_xi_ifft : test"

if true ; then
echo "matter_xi : basic r-bin tests"
./matter_xi_ifft -i ${ptcl_prefix} -o ${OUTPUTDIR}/matter_xi_ifft_00.dat
./matter_xi_ifft -i ${ptcl_prefix} --nr 100 --rrange 0 50 -o ${OUTPUTDIR}/matter_xi_ifft_01.dat
./matter_xi_ifft -i ${ptcl_prefix} --nr 100 --rrange 0 150  --log_bin -o ${OUTPUTDIR}/matter_xi_ifft_02.dat
fi

if true ; then
echo "matter_xi :  aliasing and shotnoise tests"
./matter_xi_ifft -i ${ptcl_prefix} --p_assign 1  -o ${OUTPUTDIR}/matter_xi_ifft_10.dat
./matter_xi_ifft -i ${ptcl_prefix} --p_assign 2  -o ${OUTPUTDIR}/matter_xi_ifft_11.dat
./matter_xi_ifft -i ${ptcl_prefix} --p_assign 3  -o ${OUTPUTDIR}/matter_xi_ifft_12.dat
./matter_xi_ifft -i ${ptcl_prefix} --no_shotnoise  -o ${OUTPUTDIR}/matter_xi_ifft_13.dat
fi


echo "matter_pk : test"

if true ; then
echo "matter_pk : basic k-bin tests"
./matter_pk -i ${ptcl_prefix} -o ${OUTPUTDIR}/matter_pk_00.dat
./matter_pk -i ${ptcl_prefix} --nk 100 --krange 1e-3 10 -o ${OUTPUTDIR}/matter_pk_01.dat
./matter_pk -i ${ptcl_prefix} --nk 100 --krange 1e-3 10 --log_bin -o ${OUTPUTDIR}/matter_pk_02.dat
fi
