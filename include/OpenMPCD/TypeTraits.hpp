/**
 * @file
 * Defines type traits.
 */

#ifndef OPENMPCD_TYPETRAITS_HPP
#define OPENMPCD_TYPETRAITS_HPP

#include <complex>

namespace OpenMPCD
{
	/**
	 * Contains information on certain types.
	 *
	 * The following types are defined:
	 * <table>
	 * 	<tr>
	 * 		<th>Name</th>
	 * 		<th>Meaning</th>
	 * 	</tr><tr>
	 * 		<td>RealType</td>
	 * 		<td>
	 * 			If T is a floating-point type, this is T.
	 *			If T is of the form std::complex<U> for some floating-point type
	 *			U, this is U.
	 *		</td>
	 * 	</tr>
	 * 	</tr><tr>
	 * 		<td>isStandardFloatingPoint</td>
	 * 		<td>
	 *			Whether `T` is a floating-point type, as defined in the C++
	 *			standard (i.e. `float`, `double`, or `long double`).
	 *		</td>
	 * 	</tr><tr>
	 * 		<td>isComplex</td>
	 * 		<td>
	 *			If T is of the form `std::complex<U>` for some type `U`, this
	 *			is `true`, else `false`.
	 *		</td>
	 * 	</tr>
	 * </table>
	 *
	 * @tparam T The type in question.
	 */
	template<typename T> struct TypeTraits;

	/**
	 * Specialization of `OpenMPCD::TypeTraits` for `float`.
	 *
	 * @see OpenMPCD::TypeTraits.
	 */
	template<> struct TypeTraits<float>
	{
		typedef float RealType;
			///< The type of the real (i.e. non-complex) component.

		static const bool isStandardFloatingPoint = true;
			/**< Whether this is one of the language-standard floating point
			     types.*/

		static const bool isComplex = false;
			///< Whether this type is of type `std::complex<U>` for some `U`.
	};

	/**
	 * Specialization of `OpenMPCD::TypeTraits` for `double`.
	 *
	 * @see OpenMPCD::TypeTraits.
	 */
	template<> struct TypeTraits<double>
	{
		typedef double RealType;
			///< The type of the real (i.e. non-complex) component.

		static const bool isStandardFloatingPoint = true;
			/**< Whether this is one of the language-standard floating point
			     types.*/

		static const bool isComplex = false;
			///< Whether this type is of type `std::complex<U>` for some `U`.
	};

	/**
	 * Specialization of `OpenMPCD::TypeTraits` for `long double`.
	 *
	 * @see OpenMPCD::TypeTraits.
	 */
	template<> struct TypeTraits<long double>
	{
		typedef long double RealType;
			///< The type of the real (i.e. non-complex) component.

		static const bool isStandardFloatingPoint = true;
			/**< Whether this is one of the language-standard floating point
			     types.*/

		static const bool isComplex = false;
			///< Whether this type is of type `std::complex<U>` for some `U`.
	};

	/**
	 * Specialization of `OpenMPCD::TypeTraits` for `std::complex<float>`.
	 *
	 * @see OpenMPCD::TypeTraits.
	 */
	template<> struct TypeTraits<std::complex<float> >
	{
		typedef float RealType;
			///< The type of the real (i.e. non-complex) component.

		static const bool isStandardFloatingPoint = false;
			/**< Whether this is one of the language-standard floating point
			     types.*/

		static const bool isComplex = true;
			///< Whether this type is of type `std::complex<U>` for some `U`.
	};

	/**
	 * Specialization of `OpenMPCD::TypeTraits` for `std::complex<double>`.
	 *
	 * @see OpenMPCD::TypeTraits.
	 */
	template<> struct TypeTraits<std::complex<double> >
	{
		typedef double RealType;
			///< The type of the real (i.e. non-complex) component.

		static const bool isStandardFloatingPoint = false;
			/**< Whether this is one of the language-standard floating point
			     types.*/

		static const bool isComplex = true;
			///< Whether this type is of type `std::complex<U>` for some `U`.
	};

	/**
	 * Specialization of `OpenMPCD::TypeTraits` for `std::complex<long double>`.
	 *
	 * @see OpenMPCD::TypeTraits.
	 */
	template<> struct TypeTraits<std::complex<long double> >
	{
		typedef long double RealType;
			///< The type of the real (i.e. non-complex) component.

		static const bool isStandardFloatingPoint = false;
			/**< Whether this is one of the language-standard floating point
			     types.*/

		static const bool isComplex = true;
			///< Whether this type is of type `std::complex<U>` for some `U`.
	};
}

#endif
