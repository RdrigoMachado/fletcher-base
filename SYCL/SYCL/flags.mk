# ============================================================
# SYCL/flags.mk (CORRIGIDO para AdaptiveCpp / acpp)
# ============================================================

# O compilador neste container é o AdaptiveCpp (acpp)
CC = syclcc

# Flags de otimização
CFLAGS += -O3

# --- CORREÇÃO AQUI ---
# A flag -fsycl-targets (Intel DPC++) NÃO funciona aqui.
# Deve-se usar --acpp-targets.
# Substitua 'sm_70' pela arquitetura da sua GPU se necessário (sm_60, sm_80, etc)
CFLAGS += --acpp-targets=cuda:sm_89

# Se quiser compilar também para CPU (OpenMP) como backup:
# CFLAGS += --acpp-targets=cuda:sm_70,omp

# Link automático das bibliotecas (geralmente vazio pois o syclcc resolve)
LIBS +=
