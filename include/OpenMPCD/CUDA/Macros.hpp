/**
 * @file
 * Defines preprocessor macros to make better use of CUDA.
 */

#ifndef OPENMPCD_CUDA_MACROS_HPP
#define OPENMPCD_CUDA_MACROS_HPP

#include <OpenMPCD/CUDA/Exceptions.hpp>

#ifdef __CUDACC__
	/**
	 * Denotes a function to be callable both from the Host and from a CUDA Device.
	 */
	#define OPENMPCD_CUDA_HOST_AND_DEVICE __host__ __device__

	/**
	 * Denotes a function to be callable from the Host.
	 */
	#define OPENMPCD_CUDA_HOST __host__

	/**
	 * Denotes a function to be callable from a CUDA Device.
	 */
	#define OPENMPCD_CUDA_DEVICE __device__

	/**
	 * Denotes a function to be a CUDA kernel.
	 */
	#define OPENMPCD_CUDA_GLOBAL __global__
#else
	#define OPENMPCD_CUDA_HOST_AND_DEVICE
	#define OPENMPCD_CUDA_HOST
	#define OPENMPCD_CUDA_DEVICE
	#define OPENMPCD_CUDA_GLOBAL
#endif

/**
 * Helper to be used prior to a CUDA kernel call.
 * This defines the variables gridSize and blockSize for use in the CUDA kernel call,
 * which specify the number of blocks in the grid and the number of threads in a block,
 * respectively.
 * Also, this defines the workUnitOffset variable, which can be passed as an argument to
 * the CUDA kernel. Since there may be so many work units that one grid cannot handle them
 * all, the kernel call is looped over, and the workUnitOffset tells how many work units
 * have been dispatched in previous kernel calls.
 *
 * @see OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
 * @param[in] numberOfWorkUnits_ The number of work units that need to be dispatched.
 * @param[in] maxGridSize_       The maximum number of blocks to dispatch in a grid.
 * @param[in] blockSize_         The number of threads to dispatch per block.
 */
#define OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN(numberOfWorkUnits_, maxGridSize_, blockSize_) \
	{ \
		const unsigned int _numberOfWorkUnits = (numberOfWorkUnits_); \
		static const unsigned int _maxGridSize = (maxGridSize_); \
		static const unsigned int _blockSize = (blockSize_); \
		for(unsigned int _workUnit = 0; _workUnit < _numberOfWorkUnits; _workUnit += _maxGridSize * _blockSize) \
		{ \
			const unsigned int _requiredGridSize = (_numberOfWorkUnits - _workUnit) / _blockSize + 1; \
			\
			const unsigned int gridSize = _requiredGridSize > _maxGridSize ? _maxGridSize : _requiredGridSize; \
			const unsigned int blockSize = _blockSize; \
			const unsigned int workUnitOffset = _workUnit;

/**
 * Helper to be used after a CUDA kernel call.
 * @see OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN
 */
#define OPENMPCD_CUDA_LAUNCH_WORKUNITS_END \
		} \
	}

/**
 * Calls OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN with default value for blockSize_.
 *
 * @see OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN
 * @param[in] numberOfWorkUnits_ The number of work units that need to be dispatched.
 * @param[in] maxGridSize_       The maximum number of blocks to dispatch in a grid.
 */
#define OPENMPCD_CUDA_LAUNCH_WORKUNITS_GRIDSIZE_BEGIN(numberOfWorkUnits_, maxGridSize_) \
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN((numberOfWorkUnits_), (maxGridSize_), 512)

/**
 * Calls OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN with default values for maxGridSize_ and blockSize_.
 *
 * @see OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN
 * @param[in] numberOfWorkUnits_ The number of work units that need to be dispatched.
 */
#define OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(numberOfWorkUnits_) \
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN((numberOfWorkUnits_), 1024, 512)

/**
 * Throws if the last CUDA call was not successful.
 */
#define OPENMPCD_CUDA_THROW_ON_ERROR do{const cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) OPENMPCD_THROW(OpenMPCD::CUDA::Exception, cudaGetErrorString(e));} while(0)

#endif
