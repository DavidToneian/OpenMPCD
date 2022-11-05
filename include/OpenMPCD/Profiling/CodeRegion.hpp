/**
 * @file
 * Defines the `OpenMPCD::Profiling::CodeRegion` class.
 */

#ifndef OPENMPCD_PROFILING_CODEREGION_HPP
#define OPENMPCD_PROFILING_CODEREGION_HPP

#ifdef OPENMPCD_CUDA_PROFILE
	#include <nvToolsExt.h>
#endif

namespace OpenMPCD
{
/**
 * Namespace for profiling of the application.
 */
namespace Profiling
{
	/**
	 * Marks a code region.
	 * The marked region starts with the construction of an instance of this class,
	 * and ends with its destruction.
	 */
	class CodeRegion
	{
		public:
			/**
			 * The constructor.
			 * @param[in] name The name for the code region.
			 */
			CodeRegion(const char* const name)
			{
				#ifdef OPENMPCD_CUDA_PROFILE
					nvtxRangePushA(name);
				#endif
			}

		private:
			CodeRegion(const CodeRegion&); ///< The copy constructor.

		public:
			/**
			 * The destructor.
			 */
			~CodeRegion()
			{
				#ifdef OPENMPCD_CUDA_PROFILE
					nvtxRangePop();
				#endif
			}

		private:
			const CodeRegion& operator=(const CodeRegion&); ///< The assignment operator.
	};
} //namespace Profiling
} //namespace OpenMPCD

#endif //OPENMPCD_PROFILING_CODEREGION_HPP
