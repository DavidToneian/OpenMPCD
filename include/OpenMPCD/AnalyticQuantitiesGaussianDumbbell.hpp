/**
 * @file
 * Defines the OpenMPCD::AnalyticQuantitiesGaussianDumbbell class.
 */

#ifndef OPENMPCD_ANALYTICQUANTITIESGAUSSIANDUMBBELL_HPP
#define OPENMPCD_ANALYTICQUANTITIESGAUSSIANDUMBBELL_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
	/**
	 * Computes analytic quantities for Gaussian Dumbbells.
	 * Gaussian Dumbbells are described in
	 * "Multiparticle collision dynamics simulations of viscoelastic fluids: Shear-thinning
	 * Gaussian dumbbells" by Bartosz Kowalik and Roland G. Winkler. Journal of Chemical Physics 138,
	 * 104903 (2013). http://dx.doi.org/10.1063/1.4792196
	 * The formulas used can be found there.
	 */
	class AnalyticQuantitiesGaussianDumbbell
	{
		private:
			AnalyticQuantitiesGaussianDumbbell(); ///< The constructor.

		public:
			/**
			 * Returns the zero-shear Lagrangian multiplier \f$ \lambda_0 \f$.
			 * @param[in] rootMeanSquareBondLength The root mean square bond length \f$ l \f$ of the dumbbell.
			 */
			static FP zeroShearLagrangianMultiplier(const FP rootMeanSquareBondLength)
			{
				return 1.5 / (rootMeanSquareBondLength * rootMeanSquareBondLength);
			}

			/**
			 * Returns the Lagrangian multiplier \f$ \lambda \f$.
			 * @param[in] rootMeanSquareBondLength The root mean square bond length \f$ l \f$ of the dumbbell.
			 * @param[in] shearRate                The shear rate \f$ \dot{\gamma} \f$.
			 * @param[in] zeroShearRelaxationTime  The zero-shear relaxation time \f$ \tau_0 \f$ of the dumbbell.
			 */
			static FP lagrangianMultiplier(const FP rootMeanSquareBondLength, const FP shearRate, const FP zeroShearRelaxationTime);

			/**
			 * Returns the zero-shear relaxation time \f$ \tau_0 \f$ of the dumbbell.
			 * @param[in] rootMeanSquareBondLength    The root mean square bond length \f$ l \f$ of the dumbbell.
			 * @param[in] mpcSelfDiffusionCoefficient The self-diffusion coefficient \f$ D \f$ of the bare MPC fluid
			 *                                        particle.
			 * @param[in] mpcHydrodynamicRadius       The hydrodynamic (Stokes-Einstein) radius \f$ R_H \f$ of the bare
			 *                                        MPC fluid particle.
			 */
			static FP zeroShearRelaxationTime(const FP rootMeanSquareBondLength, const FP mpcSelfDiffusionCoefficient,
			                                  const FP mpcHydrodynamicRadius);

			/**
			 * Returns the Weissenberg number \f$ \mathrm{Wi} \f$.
			 * @param[in] shearRate               The shear rate \f$ \dot{\gamma} \f$.
			 * @param[in] zeroShearRelaxationTime The zero-shear relaxation time \f$ \tau_0 \f$ of the dumbbell.
			 */
			static FP weissenbergNumber(const FP shearRate, const FP zeroShearRelaxationTime)
			{
				return shearRate * zeroShearRelaxationTime;
			}

			/**
			 * Returns the ratio \f$ \mu = \frac{\lambda}{\lambda_0} \f$ of the Lagrangian multipliers with and without shear flow.
			 * @param[in] weissenbergNumber The Weissenberg number \f$ \mathrm{Wi} \f$.
			 */
			static FP lagrangianMultiplierRatio(const FP weissenbergNumber);
	};
}

#endif
