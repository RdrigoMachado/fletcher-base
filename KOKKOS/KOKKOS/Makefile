include ../config.mk
include flags.mk

# Lista de objetos específicos do Kokkos
OBJ = kokkos_driver.o \
      kokkos_stuff.o \
      kokkos_propagate.o \
      kokkos_insertsource.o

all: $(OBJ)

# Regra genérica para .cpp -> .o
%.o: %.cpp
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.a
