CXX = g++
CXXFLAGS = -std=gnu++17 -march=native -O3 -fopenmp

HDF5_INC = -I/usr/include/hdf5/serial
HDF5_LIB = -lhdf5_serial

FFTW_INC =
FFTW_LIB = -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f

EXEC = matter_pk matter_xi matter_xi_ifft halo_pk halo_xi halo_xi_ifft

all: $(EXEC)

matter_pk: matter_pk.o
	$(CXX) $(CXXFLAGS) $< $(FFTW_LIB) -o $@

matter_xi: matter_xi.o
	$(CXX) $(CXXFLAGS) $< $(FFTW_LIB) -o $@

matter_xi_ifft: matter_xi_ifft.o
	$(CXX) $(CXXFLAGS) $< $(FFTW_LIB) -o $@

halo_pk: halo_pk.o
	$(CXX) $(CXXFLAGS) $< $(HDF5_LIB) $(FFTW_LIB) -o $@

halo_pk_cross: halo_pk_cross.o
	$(CXX) $(CXXFLAGS) $< $(HDF5_LIB) $(FFTW_LIB) -o $@

halo_xi: halo_xi.o
	$(CXX) $(CXXFLAGS) $< $(HDF5_LIB) $(FFTW_LIB) -o $@

halo_xi_cross: halo_xi_cross.o
	$(CXX) $(CXXFLAGS) $< $(HDF5_LIB) $(FFTW_LIB) -o $@

halo_xi2D: halo_xi2D.o
	$(CXX) $(CXXFLAGS) $< $(HDF5_LIB) $(FFTW_LIB) -o $@

halo_xi2D_cross: halo_xi2D_cross.o
	$(CXX) $(CXXFLAGS) $< $(HDF5_LIB) $(FFTW_LIB) -o $@

halo_xi_ifft: halo_xi_ifft.o
	$(CXX) $(CXXFLAGS) $< $(HDF5_LIB) $(FFTW_LIB) -o $@

xi_from_pk: xi_from_pk.o
	$(CXX) $(CXXFLAGS) $< $(FFTW_LIB) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< $(HDF5_INC) $(FFTW_INC)


.PHONY : clean
clean:
	rm -rf $(EXEC) *.o
