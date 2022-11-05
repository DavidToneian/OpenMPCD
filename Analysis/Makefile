COMMON_INCLUDES = -Iinclude -I../include
COMMON_DEFINES = -DOPENMPCD_DEBUG
CORRECTNESS_COMMAND_LINE_ARGUMENTS = -Wall -pedantic
COMMON_LIBRARIES = \
	-lconfig++ \
	-lboost_system \
	-lboost_filesystem \
	-lboost_program_options \
	-lboost_iostreams \
	-lboost_regex \
	-lpthread \
	-lbz2
	
COMMON_COMMAND_LINE_ARGUMENTS = \
	-std=c++11 \
	-O3 \
	$(CORRECTNESS_COMMAND_LINE_ARGUMENTS) $(COMMON_INCLUDES) $(COMMON_DEFINES)

.PHONY: default computeTransverseVelocityCorrelationFunction

default: computeTransverseVelocityCorrelationFunction

computeTransverseVelocityCorrelationFunction:
	g++ $(COMMON_COMMAND_LINE_ARGUMENTS) \
	src/computeTransverseVelocityCorrelationFunction/*.cpp \
	src/MPCDAnalysis/*.cpp \
	-o computeTransverseVelocityCorrelationFunction \
	$(COMMON_LIBRARIES)
