include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c kokkos_driver.c
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c kokkos_stuff.cu
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c kokkos_propagate.cu
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c kokkos_insertsource.cu

clean:
	rm -f *.o *.a
