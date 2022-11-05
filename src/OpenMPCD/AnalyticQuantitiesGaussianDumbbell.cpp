#include <OpenMPCD/AnalyticQuantitiesGaussianDumbbell.hpp>

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <stdexcept>

using namespace OpenMPCD;

static const FP pi = boost::math::constants::pi<FP>();

FP AnalyticQuantitiesGaussianDumbbell::lagrangianMultiplier(const FP rootMeanSquareBondLength, const FP shearRate,
                                                            const FP zeroShearRelaxationTime)
{
	const FP lambda_0 = zeroShearLagrangianMultiplier(rootMeanSquareBondLength);
	const FP mu       = lagrangianMultiplierRatio(weissenbergNumber(shearRate, zeroShearRelaxationTime));

	return lambda_0 * mu;
}

FP AnalyticQuantitiesGaussianDumbbell::zeroShearRelaxationTime(const FP rootMeanSquareBondLength, const FP mpcSelfDiffusionCoefficient,
                                                               const FP mpcHydrodynamicRadius)
{
	const FP l   = rootMeanSquareBondLength;
	const FP D   = mpcSelfDiffusionCoefficient;
	const FP R_H = mpcHydrodynamicRadius;

	const FP denominator1 = 6*D;
	const FP denominator2 = 1 - R_H / l * sqrt(6 / pi);

	return l*l / (denominator1 * denominator2);
}

FP AnalyticQuantitiesGaussianDumbbell::lagrangianMultiplierRatio(const FP weissenbergNumber)
{
	if(weissenbergNumber==0)
		return 1; //if weissenbergNumber is 0, this means that the shear rate is 0, in which case we do not want the ratio 0, but 1.

	double x, y, z;

	const double c = - weissenbergNumber * weissenbergNumber / 6.0;

	const int rootCount = gsl_poly_solve_cubic(-1, 0, c, &x, &y, &z);

	if(rootCount!=1)
		throw std::runtime_error("Unexpected number of roots in AnalyticQuantitiesGaussianDumbbell::lagrangianMultiplierRatio");

	return x;
}
