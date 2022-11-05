#include <OpenMPCD/Utility/CompilerDetection.hpp>

#include <OpenMPCDTest/external/catch.hpp>

#ifdef OPENMPCD_COMPILER_GCC
	//gcc emits warnings when using the Catch framework macros, as in
	//    REQUIRE(val == 1);
	#pragma GCC diagnostic ignored "-Wparentheses"
#endif
