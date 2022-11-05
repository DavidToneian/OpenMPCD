#! /usr/bin/python

import argparse

import MPCDAnalysis.AnalyticQuantities_SimpleFluid as Analytic

if __name__ == '__main__':
	programDescription = 'Computes the analytical prediction for the SRD kinematic viscosity.'
	commandLineParser = argparse.ArgumentParser(description=programDescription, add_help=True)
	commandLineParser.add_argument(
		'--kT',
		help='The product of the Boltzmann constant and the temperature',
		type=float,
		dest='kT',
		default=1)
	commandLineParser.add_argument(
		'--meanParticleCountPerCell',
		help='The mean number of MPC particles per cell',
		type=int,
		dest='meanParticleCountPerCell',
		default=10)
	commandLineParser.add_argument(
		'--m',
		help='The MPC particle mass',
		type=float,
		dest='m',
		default=1)
	commandLineParser.add_argument(
		'--MPCTimestep',
		help='The MPC timestep',
		type=float,
		dest='MPCTimestep',
		default=0.1)
	commandLineParser.add_argument(
		'--SRDAngle',
		help='The SRD rotation angle.',
		type=float,
		dest='SRDAngle',
		default=2.27)
	commandLineArguments = commandLineParser.parse_args()


	kT = commandLineArguments.kT
	meanParticleCountPerCell = commandLineArguments.meanParticleCountPerCell
	m = commandLineArguments.m
	MPCTimestep = commandLineArguments.MPCTimestep
	SRDAngle = commandLineArguments.SRDAngle


	if kT <= 0:
		print("--kT is invalid.")
		exit(1)

	if meanParticleCountPerCell <= 0:
		print("--meanParticleCountPerCell is invalid.")
		exit(1)

	if m <= 0:
		print("--m is invalid.")
		exit(1)

	if MPCTimestep <= 0:
		print("--MPCTimestep is invalid.")
		exit(1)

	if SRDAngle <= 0:
		print("--SRDAngle is invalid.")
		exit(1)

	print("kT:\t" + str(kT))
	print("meanParticleCountPerCell:\t" + str(meanParticleCountPerCell))
	print("m:\t" + str(m))
	print("MPCTimestep:\t" + str(MPCTimestep))
	print("SRDAngle:\t" + str(SRDAngle))

	print("")


	kineticContribution = Analytic.kineticContributionsToSRDKinematicShearViscosity(meanParticleCountPerCell, kT, m, MPCTimestep, SRDAngle)
	collisionalContribution = Analytic.collisionalContributionsToSRDKinematicShearViscosity(1, meanParticleCountPerCell, MPCTimestep, SRDAngle)
	total = kineticContribution + collisionalContribution

	print("Kinetic Contribution:\t" + str(kineticContribution))
	print("Collisional Contribution:\t" + str(collisionalContribution))
	print("Total:\t" + str(total))