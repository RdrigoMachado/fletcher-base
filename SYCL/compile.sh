#!/usr/bin/env bash
set -e  # Para parar em caso de erro

echo "=== Carregando ambiente do Kokkos ==="
spack load kokkos
spack load kokkos-nvcc-wrapper

# Descobre o caminho de instalação
KOKKOS_PATH=$(spack location -i kokkos)
echo "Kokkos instalado em: $KOKKOS_PATH"

# Define o compilador como o nvcc_wrapper do Kokkos
export CXX="${KOKKOS_PATH}/bin/nvcc_wrapper"

# Inclui os diretórios de include e lib
export CXXFLAGS="-I${KOKKOS_PATH}/include"
export LDFLAGS="-L${KOKKOS_PATH}/lib -lkokkoscore"

echo "=== Compilando com Make ==="
make KOKKOS_PATH="${KOKKOS_PATH}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS}" LDFLAGS="${LDFLAGS}"

echo "✅ Build finalizado com sucesso!"
