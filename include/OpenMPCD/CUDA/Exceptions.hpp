/**
 * @file
 * Defines exceptions for CUDA programs.
 */

#ifndef OPENMPCD_CUDA_EXCEPTIONS_HPP
#define OPENMPCD_CUDA_EXCEPTIONS_HPP

#include <OpenMPCD/Exceptions.hpp>

namespace OpenMPCD
{
	namespace CUDA
	{
		/**
		 * Base CUDA exception.
		 */
		class Exception : public OpenMPCD::Exception
		{
			public:
				/**
				 * The constructor.
				 * @param[in] msg The error message.
				 */
				Exception(const std::string& msg)
					: OpenMPCD::Exception(msg)
				{
				}
		};
	}
}

#endif
