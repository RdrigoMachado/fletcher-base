include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c kokkos_driver.cpp
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c kokkos_stuff.cpp
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c kokkos_propagate.cpp
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c kokkos_insertsource.cpp

clean:
	rm -f *.o *.a
