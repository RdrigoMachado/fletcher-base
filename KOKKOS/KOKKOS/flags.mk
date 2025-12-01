# ==============================================================================
# KOKKOS/flags.mk - Configuração para compilar dentro do Container Singularity
# ==============================================================================

# O container define a variável de ambiente KOKKOS_ROOT.
# Se não estiver definida (fora do container), usa um fallback (opcional).
KOKKOS_PATH ?= $(KOKKOS_ROOT)

ifeq ($(KOKKOS_PATH),)
    # Fallback caso tente compilar fora do container sem carregar módulo
    KOKKOS_PATH = /opt/kokkos/4.3.00
endif

# O Wrapper do NVCC é essencial para separar código Host/Device
CC = $(KOKKOS_PATH)/bin/nvcc_wrapper

# Flags de compilação
# --expt-extended-lambda é vital para CUDA lambdas no Kokkos
CFLAGS += -O3 -std=c++17
CFLAGS += --expt-extended-lambda
CFLAGS += -I$(KOKKOS_PATH)/include

# --- CORREÇÃO AQUI ---
# O Kokkos do container exige OpenMP.
# Usamos -Xcompiler para garantir que o nvcc passe a flag para o gcc
CFLAGS += -Xcompiler -fopenmp
# ---------------------

# Flags de linkagem
# Também precisamos linkar com a lib do OpenMP (gomp)
LIBS += -L$(KOKKOS_PATH)/lib -L$(KOKKOS_PATH)/lib64 -lkokkoscore -lgomp -lcuda -lcudart
