backend ?= SYCL
arch = $(backend)

include config.mk
include $(arch)/flags.mk

OBJ1 = \
	source.o \
	utils.o \
	boundary.o \
	walltime.o \
	model.o \
	map.o

ifdef PAPI
	LIBS += $(PAPI_LIBS)
	OBJ1 += ModPAPI.o
	CFLAGS += -DPAPI
endif

OBJ = main.o $(OBJ1)

TARGET = ModelagemFletcher.exe

all: $(OBJ)
	cd $(arch) && make
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -o $(TARGET) $(OBJ) $(arch)/*.o $(LIBS)

main.o: main.cpp $(OBJ1)
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) main.cpp

boundary.o: boundary.cpp boundary.h map.o
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) boundary.cpp

source.o: source.cpp source.h
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) source.cpp

utils.o: utils.cpp utils.h map.o source.o
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) utils.cpp

map.o: map.cpp map.h
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) map.cpp

model.o: model.cpp model.h
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) model.cpp

walltime.o: walltime.cpp walltime.h
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) walltime.cpp

ModPAPI.o: ModPAPI.cpp ModPAPI.h
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) ModPAPI.cpp

clean:
	cd $(arch) && make clean
	rm -f *.o $(TARGET)

clean-all:
	rm -f */*.o *.o $(TARGET)
