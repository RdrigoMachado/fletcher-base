# Environment and compilation flags at Exaflop
#   module load gompi

# Default
backend=KOKKOS
arch=$(backend)

include config.mk
include $(arch)/flags.mk

OBJ1=\
	source.o \
	utils.o \
	boundary.o \
	walltime.o \
	model.o

ifdef PAPI
	LIBS += $(PAPI_LIBS)
	OBJ1 += ModPAPI.o
	CFLAGS += -DPAPI
endif

OBJ=main.o $(OBJ1)

TARGET = ModelagemFletcher.exe

all: 	$(OBJ)
	cd $(arch) && make
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(arch)/*.o $(LIBS)

main.o:	main.cpp $(OBJ1)
	$(CC) -c $(CFLAGS) main.cpp

boundary.o:	boundary.cpp boundary.h
	$(CC) -c $(CFLAGS) boundary.cpp

source.o:	source.cpp source.h
	$(CC) -c $(CFLAGS) source.cpp

utils.o:	utils.cpp utils.h
	$(CC) -c $(CFLAGS) utils.cpp

model.o:	model.cpp model.h
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) model.cpp

walltime.o:	walltime.cpp walltime.h
	$(CC) -c $(CFLAGS) walltime.cpp

ModPAPI.o:	ModPAPI.cpp ModPAPI.h
	$(CC) -c $(CFLAGS) ModPAPI.cpp


.SUFFIXES	:	.o .cpp

.cpp.o:
	$(CC) $(CFLAGS) $*.cpp

clean:
	cd $(arch) && make clean
	rm -f *.o $(TARGET)

clean-all:
	rm -f */*.o *.o $(TARGET)
