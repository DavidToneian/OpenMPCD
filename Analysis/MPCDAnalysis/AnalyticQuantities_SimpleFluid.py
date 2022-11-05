import math

def kineticContributionsToSRDKinematicShearViscosity(
									meanParticleCountPerCell, kT, m,
									mpcTimestep, srdAngle):
	M = meanParticleCountPerCell

	factor1 = kT * mpcTimestep / (2 * m)
	denominator1 = M - 1 + math.exp(-M)
	denominator2 = 2 - math.cos(srdAngle) - math.cos(2 * srdAngle)
	factor2 = 5 * M / (denominator1 * denominator2) - 1

	return factor1 * factor2;

def collisionalContributionsToSRDKinematicShearViscosity(
		linearCellSize, meanParticleCountPerCell, mpcTimestep, srdAngle):
	"""
	Returns the collisional contribution to the kinematic shear viscosity in SRD,
	according to table 1 in
	Gompper, Ihle, Kroll, and Winkler:
	"Multi-Particle Collision Dynamics: A Particle-Based Mesoscale Simulation Approach to the Hydrodynamics of Complex Fluids"
	DOI: 10.1007/12_2008_5

	@param[in] linearCellSize           The size of a collision cell. This is called "a" in the reference, and is usually 1.
	@param[in] meanParticleCountPerCell The average number of MPC particles per collision cell.
	@param[in] mpcTimestep              The timestep for the MPC streaming step.
	@param[in] srdAngle                 The rotation angle in SRD.
	"""
	M = meanParticleCountPerCell

	factor1 = linearCellSize * linearCellSize / (6 * 3 * mpcTimestep);
	factor2 = (M - 1 + math.exp(-M)) / M;
	factor3 = 1 - math.cos(srdAngle);

	return factor1 * factor2 * factor3;

def approximateSelfDiffusionCoefficient(
									meanParticleCountPerCell, m,
									kT, mpcTimestep, dimensions, srdAngle):
	M = meanParticleCountPerCell

	factor1 = kT * mpcTimestep / (2.0 * m)
	term2 = dimensions * M / ((1 - math.cos(srdAngle)) * (M - 1 + math.exp(-M)))

	return factor1 * (term2 - 1)
