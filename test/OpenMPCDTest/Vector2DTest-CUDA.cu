/**
 * @file
 * Tests `OpenMPCD::Vector2D` on CUDA Devices.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>


#define TEST_DATATYPES (float)(double)(OpenMPCD::FP)
#define TEST_DATATYPES_HIGHPRECISION (double)(OpenMPCD::FP)


#define OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION __global__

#define OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG bool* const _result

#define OPENMPCDTEST_VECTOR2DTEST_SCENARIO_SUFFIX ", CUDA"

#define OPENMPCDTEST_VECTOR2DTEST_SCENARIO_TAG "[CUDA]"

#define REQ(expr) if(!(expr)) *_result = false;

#define REQ_FALSE(expr) if(expr) *_result = false;

#define OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTIONS_CALL(funcReal, funcComplex) \
	do \
	{ \
		static const bool _true_ = true; \
		\
		OpenMPCD::CUDA::DeviceMemoryManager _dmm; \
		_dmm.setAutofree(true); \
		\
		bool* _result; \
		_dmm.allocateMemory(&_result, 1); \
		_dmm.copyElementsFromHostToDevice(&_true_, _result, 1); \
		\
		funcReal<<<1, 1>>>(_result); \
		cudaDeviceSynchronize(); \
		OPENMPCD_CUDA_THROW_ON_ERROR; \
		\
		REQUIRE(_dmm.elementMemoryEqualOnHostAndDevice(&_true_, _result, 1)); \
	} while(0)


#include "Vector2DTest.cpp"
