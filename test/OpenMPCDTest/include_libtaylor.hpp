/**
 * @file
 * Includes `OpenMPCDTest/external/libtaylor/taylor.hpp` and defines suitable
 * specializations for `OpenMPCD::TypeTraits`.
 */

#ifndef OPENMPCDTEST_INCLUDE_LIBTAYLOR_HPP
#define OPENMPCDTEST_INCLUDE_LIBTAYLOR_HPP

#include <OpenMPCDTest/external/libtaylor/taylor.hpp>

#include <OpenMPCD/Scalar.hpp>
#include <OpenMPCD/TypeTraits.hpp>

namespace OpenMPCD
{

namespace Scalar
{
template<typename T, int variableCount, int order>
OPENMPCD_CUDA_HOST_AND_DEVICE
taylor<T, variableCount, order>
getRealPart(const taylor<T, variableCount, order>& val)
{
	return val;
}

template<typename T, int variableCount, int order>
OPENMPCD_CUDA_HOST_AND_DEVICE
bool isZero(const taylor<T, variableCount, order>& val)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

	for(unsigned int i=0; i < val.size; ++i)
	{
		if(val.c[i] != 0)
			return false;
	}

	return true;
}
} //namespace Scalar

template<typename T, int variableCount, int order>
struct TypeTraits<taylor<T, variableCount, order> >
{
	typedef
		typename boost::enable_if<
			boost::is_floating_point<T>,
			taylor<T, variableCount, order> >::type
		RealType;
};

} //namespace OpenMPCD


#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{
namespace Implementation_Vector3D
{
template<typename T, int variableCount, int order>
class Dot<
	taylor<T, variableCount, order>,
	typename boost::enable_if<boost::is_floating_point<T> >::type>
{
	private:
	Dot();

	public:
	static
	OPENMPCD_CUDA_HOST_AND_DEVICE
	const taylor<T, variableCount, order> dot(
		const Vector3D<taylor<T, variableCount, order> >& lhs,
		const Vector3D<taylor<T, variableCount, order> >& rhs)
	{
		BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

		return
			lhs.getX()*rhs.getX() +
			lhs.getY()*rhs.getY() +
			lhs.getZ()*rhs.getZ();
	}
}; //class Dot, partial template specialization for
   //taylor<T, variableCount, order> with T being a floating point type
} //namespace Implementation_Vector3D
} //namespace OpenMPCD

template<typename T, int variableCount, int order>
bool operator>(
	const taylor<T, variableCount, order>& lhs,
	const taylor<T, variableCount, order>& rhs)
{
	return lhs[0] > rhs[0];
}
template<typename T, int variableCount, int order>
bool operator>=(
	const taylor<T, variableCount, order>& lhs,
	const taylor<T, variableCount, order>& rhs)
{
	return lhs[0] >= rhs[0];
}

#endif //OPENMPCDTEST_INCLUDE_LIBTAYLOR_HPP
