__usedOnlySoTheRightHandSideCommandIsInvoked := $(shell tools/writeVersionFile.sh )

SOURCES = $(shell find src \( ! -path 'src/OpenMPCD/main.cpp' \( -name '*.cpp' \) \) -print )
SOURCESCU = $(shell find src \( ! -path 'src/OpenMPCD/main.cpp'  \( -name '*.cu' \) \) -print )
MAIN = src/OpenMPCD/main.cpp
TEST_SOURCES = $(shell find test/OpenMPCDTest \( ! -path 'src/OpenMPCD/main.cpp'  \( -name '*.cpp' \) \) -print )
TEST_SOURCESCU = $(shell find test/OpenMPCDTest \( ! -path 'src/OpenMPCD/main.cpp'  \( -name '*.cu' \) \) -print )

BUILDDIR := build
DEPDIR := $(BUILDDIR)/dep
TESTDEPDIR := $(BUILDDIR)/testdep

	  

EXECUTABLE = openmpcd
TESTEXECUTABLE = test/runtests
OBJECTSCPP = $(patsubst %.cpp, $(BUILDDIR)/obj/%.o, $(SOURCES))
OBJECTMAIN = $(patsubst %.cpp,$(BUILDDIR)/obj/%.o, $(MAIN))
OBJECTSCU = $(patsubst %.cu,$(BUILDDIR)/obj/%.o, $(SOURCESCU))

OBJECTSTEST = $(patsubst %.cpp, $(BUILDDIR)/obj/%.o, $(TEST_SOURCES))
OBJECTSTESTCU = $(patsubst %.cu, $(BUILDDIR)/obj/%.o, $(TEST_SOURCESCU))

SUPPRESS_DEPRECATED_GPU_TARGET_WARNING = -Wno-deprecated-gpu-targets
CUDA_ARCH = -arch=sm_20

NVCC_BOOST_WORKAROUND = -DBOOST_NOINLINE='__attribute__ ((noinline))'

NVCC_WORKAROUNDS = $(NVCC_BOOST_WORKAROUND)
#nvcc version 7.5, at least on Ubuntu 16.04, complains about `memcpy` not being defined in `string.h` without this:
ifeq ($(shell nvcc --version | grep -q -s V7.5 && echo yes)$(shell grep -q -s "Ubuntu 16.04" /etc/issue && echo yes),yesyes)
	NVCC_WORKAROUNDS += -D_FORCE_INLINES
endif

GCC_MAJOR_VERSION_GREATER_THAN_5 := $(shell expr `gcc -dumpversion | cut -f1 -d.` \> 5)
CUDA_MAJOR_VERSION := $(shell nvcc --version | grep -Po '(?<=V)(\d+)')
CUDA_MAJOR_VERSION_LESS_THAN_9 := $(shell expr $(CUDA_MAJOR_VERSION) \< 9)
ifeq "$(GCC_MAJOR_VERSION_GREATER_THAN_5)$(CUDA_MAJOR_VERSION_LESS_THAN_9)" "11"
	NVCC_WORKAROUNDS += -ccbin=$(shell which g++-5)
endif

COMMON_INCLUDES = -Iinclude -IcudaDeviceCode
TEST_INCLUDES = -Itest
COMMON_DEFINES = -DOPENMPCD_DEBUG -DOPENMPCD_CUDA_DEBUG
COMMON_COMMAND_LINE_ARGUMENTS = $(COMMON_INCLUDES) $(COMMON_DEFINES)


MPIFLAGS_COMPILE := $(shell mpic++ --showme:compile | sed 's/-pthread/-Xcompiler -pthread/')
MPIFLAGS_LINK := $(shell mpic++ --showme:link | sed 's/-pthread/-lpthread/g' | sed 's/-Wl,/-Xlinker /g')

GENERATEDEPS = ./generateDependencies $@ $< "$(COMMON_INCLUDES) $(MPIFLAGS_COMPILE) $(SUPPRESS_DEPRECATED_GPU_TARGET_WARNING)" > $(DEPDIR)/$*.Td
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

GENERATETESTDEPS = ./generateDependencies $@ $< "$(COMMON_INCLUDES) $(TEST_INCLUDES) $(MPIFLAGS_COMPILE) $(SUPPRESS_DEPRECATED_GPU_TARGET_WARNING)" > $(TESTDEPDIR)/$*.Td
POSTTESTCOMPILE = mv -f $(TESTDEPDIR)/$*.Td $(TESTDEPDIR)/$*.d

NVCC_COMMAND_LINE_ARGUMENTS = $(NVCC_WORKAROUNDS) -Xcompiler -Wall $(CUDA_ARCH) $(SUPPRESS_DEPRECATED_GPU_TARGET_WARNING) --use_fast_math

LIBRARIES = -lconfig++ -lgsl -lgslcblas -lboost_system -lboost_filesystem -lboost_program_options -lboost_regex -lboost_chrono -lboost_thread
LIBRARIES_TEST = -lboost_regex



ifdef PROFILE
	LIBRARIES += -lnvToolsExt
	COMMON_DEFINES += -DOPENMPCD_CUDA_PROFILE
endif

$(shell mkdir -p $(BUILDDIR) >/dev/null)
$(shell mkdir -p $(DEPDIR) >/dev/null)
$(shell mkdir -p $(TESTDEPDIR) >/dev/null)
$(shell mkdir -p $(BUILDDIR)/obj >/dev/null)

NVCC = nvcc

.PHONY: clean

all: $(EXECUTABLE) $(TESTEXECUTABLE)

$(EXECUTABLE): $(OBJECTSCPP) $(OBJECTSCU) $(OBJECTMAIN)
	@echo building executable $@
	@$(NVCC) $(CUDA_ARCH) --device-link -o $(BUILDDIR)/device-code.o $(OBJECTSCU) $(OBJECTMAIN)
	@$(NVCC) $(COMMON_COMMAND_LINE_ARGUMENTS) $(NVCC_COMMAND_LINE_ARGUMENTS) $(MPIFLAGS_LINK) -g -G -O3 -o $@ $(BUILDDIR)/device-code.o $(OBJECTSCPP) $(OBJECTSCU) $(OBJECTMAIN) $(LIBRARIES)

$(TESTEXECUTABLE): $(OBJECTSTEST) $(OBJECTSTESTCU) $(OBJECTSCPP) $(OBJECTSCU)
	@echo building executable $@
	@$(NVCC) $(CUDA_ARCH) --device-link -o $(BUILDDIR)/device-code-test.o $(OBJECTSCU) $(OBJECTSTESTCU)
	@$(NVCC) $(COMMON_COMMAND_LINE_ARGUMENTS) -DOPENMPCD_CUDA_DEBUG $(NVCC_COMMAND_LINE_ARGUMENTS) $(MPIFLAGS_LINK) \
	-g -G -O0 -o test/runtests $(BUILDDIR)/device-code-test.o $(OBJECTSTEST) $(OBJECTSTESTCU) $(OBJECTSCPP) $(OBJECTSCU) $(LIBRARIES) $(LIBRARIES_TEST)
	
$(OBJECTMAIN): $(BUILDDIR)/obj/%.o: %.cpp $(DEPDIR)/%.d
	@echo building object file $@
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)/$*.d)
	@$(GENERATEDEPS)
	@$(NVCC) $(COMMON_COMMAND_LINE_ARGUMENTS) $(NVCC_COMMAND_LINE_ARGUMENTS) $(MPIFLAGS_COMPILE) --device-c $< -g -G -O3 -o $@
	@$(POSTCOMPILE)

$(OBJECTSCPP): $(BUILDDIR)/obj/%.o: %.cpp $(DEPDIR)/%.d
	@echo building object file $@
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)/$*.d)
	@$(GENERATEDEPS)
	@$(NVCC) $(COMMON_COMMAND_LINE_ARGUMENTS) $(NVCC_COMMAND_LINE_ARGUMENTS) $(MPIFLAGS_COMPILE) -c $< -g -G -O3 -o $@
	@$(POSTCOMPILE)

$(OBJECTSCU): $(BUILDDIR)/obj/%.o: %.cu $(DEPDIR)/%.d
	@echo building object file $@
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)/$*.d)
	@$(GENERATEDEPS)
	@$(NVCC) $(COMMON_COMMAND_LINE_ARGUMENTS) $(NVCC_COMMAND_LINE_ARGUMENTS) $(MPIFLAGS_COMPILE) --device-c $< -g -G -O3 -o $@
	@$(POSTCOMPILE)
		
$(OBJECTSTEST): $(BUILDDIR)/obj/%.o: %.cpp $(TESTDEPDIR)/%.d
	@echo building object file $@
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(TESTDEPDIR)/$*.d)
	@$(GENERATETESTDEPS)
	@$(NVCC) $(COMMON_COMMAND_LINE_ARGUMENTS) -DOPENMPCD_CUDA_DEBUG $(NVCC_COMMAND_LINE_ARGUMENTS) $(MPIFLAGS_COMPILE) $(TEST_INCLUDES) -c $< -g -G -O0 -o $@
	@$(POSTTESTCOMPILE)
	
$(OBJECTSTESTCU): $(BUILDDIR)/obj/%.o: %.cu $(TESTDEPDIR)/%.d
	@echo building object file $@
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(TESTDEPDIR)/$*.d)
	@$(GENERATETESTDEPS)
	@$(NVCC) $(COMMON_COMMAND_LINE_ARGUMENTS) -DOPENMPCD_CUDA_DEBUG $(NVCC_COMMAND_LINE_ARGUMENTS) $(MPIFLAGS_COMPILE) $(TEST_INCLUDES) --device-c $< -g -G -O0 -o $@
	@$(POSTTESTCOMPILE)
	
$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
$(TESTDEPDIR)/%.d: ;
.PRECIOUS: $(TESTDEPDIR)/%.d

clean:
	rm -rf $(BUILDDIR) $(EXECUTABLE) $(TESTEXECUTABLE)
	
-include $(shell find $(DEPDIR) -iname '*.d')
-include $(shell find $(TESTDEPDIR) -iname '*.d')
