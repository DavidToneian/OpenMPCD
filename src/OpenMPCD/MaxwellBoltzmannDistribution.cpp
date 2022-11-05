#include <OpenMPCD/MaxwellBoltzmannDistribution.hpp>

#include <boost/random/uniform_01.hpp>
#include <cmath>

template<typename RNG>
	OpenMPCD::FP OpenMPCD::MaxwellBoltzmannDistribution::getRandomMaxwell(const FP m, const FP kT, RNG& rng)
{
	/*
	 * Algorithm according to the "Hand-book on Statistical Distributions for experimentalists" by Christian Walck,
	 * SUF–PFY/96–01
	 */

	static boost::random::uniform_01<FP> dist;

	const FP xi_1=dist(rng);
	FP r=-log(xi_1);

	FP w_1;
	FP w;

	for(;;)
	{
		const FP xi_2=dist(rng);
		const FP xi_3=dist(rng);

		w_1=xi_2*xi_2;
		const FP w_2=xi_3*xi_3;
		w=w_1+w_2;

		if(w<1 && w!=0)
			break;
	}

	r=r-w_1/w*log(dist(rng));

	return sqrt(2*r*kT/m);
}
