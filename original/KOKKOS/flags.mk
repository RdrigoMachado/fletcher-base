# Caminho da instalação do Kokkos
KOKKOS_PATH ?= /caminho/para/kokkos

# Compilador (pode ser mpicxx se você usar MPI + Kokkos)
CC = $(KOKKOS_PATH)/bin/nvcc_wrapper  # ou mpicxx se CPU-only

# Flags de compilação
CFLAGS += -O3 -std=c++17
CFLAGS += --expt-extended-lambda
CFLAGS += -I$(KOKKOS_PATH)/include

# Flags de linkagem
LIBS += -L$(KOKKOS_PATH)/lib -lkokkoscore
