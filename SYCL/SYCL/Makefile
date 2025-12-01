# ============================================================
# SYCL/Makefile — Compilação do backend SYCL
# ============================================================

include ../config.mk
include flags.mk

OBJ = sycl_driver.o \
      sycl_stuff.o \
      sycl_propagate.o \
      sycl_insertsource.o

all: $(OBJ)

%.o: %.cpp
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.a
