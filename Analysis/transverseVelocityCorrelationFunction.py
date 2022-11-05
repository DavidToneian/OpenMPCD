#! /usr/bin/python

"""
Calculates the normalized transverse velocity correlation function :math:`C_v^T\\left(t\\right)`.

:math:`C_v^T\\left(t\\right)` is defined in equation (39) of the paper
"Hydrodynamic correlations in multiparticle collision dynamics fluids" by
Chein-Cheng Huang, Gerhard Gompper, and Roland G. Winkler,
Phys. Rev. E 86, 056711 (2012), DOI: 10.1103/PhysRevE.86.056711
as such:
.. math::
	C_v^T\\left(t\\right) = \\frac{\\left<\\vec{v}^T\\left(\\vec{k}, t\\right) \\cdot \\vec{v}^T\\left(-\\vec{k}, 0\\right)\\right>}
	                              {\\left<\\vec{v}^T\\left(\\vec{k}, 0\\right) \\cdot \\vec{v}^T\\left(-\\vec{k}, 0\\right)\\right>}

Here, the scalar dot product is not meant to include complex conjugation of either argument.
"""

from MPCDAnalysis.Data2D import Data2D
import MPCDAnalysis.InteractivePlotter.Plotter
import MPCDAnalysis.PlotTools as PlotTools
from MPCDAnalysis.Vector3D import Vector3D
import MPCDAnalysis.Utilities as Utilities

import argparse
import glob
import math
import numpy
import os.path
from pylibconfig import Config
import scipy.optimize
import sys


def getSimboxByRundir(rundir):
	config = Config()
	config.readFile(rundir + "/config.txt")
	simBoxX = config.value("mpc.simulationBoxSize.x")[0]
	simBoxY = config.value("mpc.simulationBoxSize.y")[0]
	simBoxZ = config.value("mpc.simulationBoxSize.z")[0]

	return [simBoxX, simBoxY, simBoxZ]


def getSimboxByRundirs(rundirs):
	simBox = None
	for rundir in rundirs:
		current = getSimboxByRundir(rundir)

		if simBox is None:
			simBox = current
		else:
			if simBox != current:
				raise ValueError("Incompatible runs: simulation box size mismatch")

	return simBox


def getWaveVectorIntegerComponentsByRundir(rundir):
	waveVectors = []

	for filename in sorted(glob.glob(rundir + "/fourierTransformedVelocities/*.data*")):
		with Utilities.openPossiblyCompressedFile(filename) as file:
			line = file.readline()
			if line[0] != '#':
				raise ValueError("Malformed data file: k_n vector missing")

			n = [float(x) for x in line.split()[1:]]
			k_n = Vector3D(n[0], n[1], n[2])

			waveVectors.append((k_n, os.path.basename(filename)))

	return waveVectors


def getWaveVectorIntegerComponentsByRundirs(rundirs):
	waveVectors = None
	for rundir in rundirs:
		current = getWaveVectorIntegerComponentsByRundir(rundir)

		if waveVectors is None:
			waveVectors = current
		else:
			if waveVectors != current:
				raise ValueError("Incompatible runs: wave vector mismatch")

	return waveVectors


def getWaveVectorFromIntegerComponents(k_n, simBox):
	tmp = []
	for i in range(0, 3):
		tmp.append(2 * math.pi * k_n[i] / simBox[i])

	return Vector3D(*tmp)


def processFile(path, k_n):
	data = Data2D()

	with Utilities.openPossiblyCompressedFile(path) as file:
		file.readline() #skip comment line that gives the integer components k_n
		for line in file:
			columns = [float(x) for x in line.split()]

			if len(columns) != 7:
				raise ValueError("Malformed data file")

			t = columns[0]
			vx = complex(columns[1], columns[2])
			vy = complex(columns[3], columns[4])
			vz = complex(columns[5], columns[6])
			v = Vector3D(vx, vy, vz)
			vT = v.getPerpendicularTo(k_n)

			data.addPoint(t, vT)

	return data




def writePlotfiles(rundirs, commandLineArguments, pathTemplate):
	"""
	Processes the given run directories and writes the corresponding plotfiles.

	@param[in] rundirs              A list of run directories to process.
	@param[in] commandLineArguments The command line arguments passed to this script.
	@param[in] pathTemplate         A template string for the plotfile paths,
	                                in which the integer components of the k-vector
	                                are substituted.
	"""

	simbox = getSimboxByRundirs(rundirs)
	k_n_withFilenames = getWaveVectorIntegerComponentsByRundirs(rundirs)

	kT = Utilities.getConsistentConfigValue(rundirs, 'bulkThermostat.targetkT')

	for k_n, filename in k_n_withFilenames:
		strides = {}
		for rundir in rundirs:
			data = processFile(rundir + "/fourierTransformedVelocities/" + filename, k_n)

			dataItems = data.getData().items()

			for stride in range(0, 5000):
				sum_ = complex(0, 0)
				count_ = 0
				deltaT = None

				for key, value in enumerate(dataItems):
					if key + stride >= len(dataItems):
						break

					#since Vector3D does complex conjugation, there is no need to do it manually here:
					sum_ += dataItems[key + stride][1].dot(value[1])
					count_ += 1

					if deltaT is None:
						deltaT = dataItems[key + stride][0] - value[0]

				if deltaT is None:
					break

				if stride not in strides:
					strides[stride] = {'deltaT': deltaT, 'sum': sum_, 'count': count_}
				else:
					if strides[stride]['deltaT'] != deltaT:
						raise ValueError("Incompatible timesteps")
					strides[stride]['sum'] += sum_
					strides[stride]['count'] += count_


		C_vT = Data2D()
		denominator = None
		for stride, data in strides.items():
			average = data['sum'] / data['count']

			if stride == 0:
				denominator = average

			if denominator is None:
				raise ValueError("Found no 0-stride")

			value = average / denominator
			C_vT.addPoint(data['deltaT'], value.real)

		C_vT_fitFunction = lambda x, param: numpy.exp(-param * x)
		C_vT_x, C_vT_y = C_vT.getKeysAndValues()
		fit = scipy.optimize.curve_fit(C_vT_fitFunction, numpy.array(C_vT_x), numpy.array(C_vT_y))

		k = getWaveVectorFromIntegerComponents(k_n, simbox)
		nu = fit[0][0] / k.getLengthSquared()

		plotfilePath = pathTemplate.format(k_n.getX().real, k_n.getY().real, k_n.getZ().real)
		with open(plotfilePath, 'w') as plotfile:
			plotfile.write('#rundirs = [')
			_isFirstRundir = True
			for _rundir in rundirs:
				if not _isFirstRundir:
					plotfile.write(', ')

				_isFirstRundir = False

				plotfile.write("'")
				plotfile.write(_rundir)
				plotfile.write("'")

			plotfile.write(']\n')

			plotfile.write('#k_nx = {}\n'.format(k_n.getX().real))
			plotfile.write('#k_ny = {}\n'.format(k_n.getY().real))
			plotfile.write('#k_nz = {}\n'.format(k_n.getZ().real))
			plotfile.write('#L_x = {}\n'.format(simbox[0]))
			plotfile.write('#L_y = {}\n'.format(simbox[1]))
			plotfile.write('#L_z = {}\n'.format(simbox[2]))
			plotfile.write('#kT = {:.5f}\n'.format(kT))
			plotfile.write('#nu = {:.5f}\n'.format(nu))

			C_vT.writeTo(plotfile)

		#extrema = C_vT.getLocalExtrema(True)
		#for x, y in extrema.items():
		#	if y < 0:
		#		extrema[x] = -y

		#print(zip(*extrema.items()))
		#plotX, plotY = zip(*extrema.items())
		#plt.gca().plot(plotX, plotY)



if __name__ == '__main__':
	programDescription = "Calculates the normalized transverse velocity correlation function C_v^T(t)"
	commandLineParser = argparse.ArgumentParser(description=programDescription, add_help=True)
	commandLineParser.add_argument('rundirs', help='Run directories to analyze', nargs='*')
	commandLineParser.add_argument(
		'--saveFilename', help='Filename, relative to the default directory, to save the processed data to',
		type=str, dest='saveFilename')
	commandLineParser.add_argument(
		'--dataDirectory', help='Directory in which to find simulation data files to plot',
		type=str, dest='dataDirectory')
	commandLineArguments = commandLineParser.parse_args()


	thisDirectory = os.path.dirname(os.path.realpath(__file__))
	plotfileDirectory = thisDirectory + '/../../Plotfiles/transverseVelocityCorrelationFunction/'
	theoryDirectory = thisDirectory + '/../../Calculations/data/velocityAutocorrelationInFourierSpace'


	if commandLineArguments.saveFilename is not None:
		rundirs = commandLineArguments.rundirs
		if len(rundirs) == 0:
			print('Must specify at least one rundir to save plot files.')
			exit(1)

		pathTemplate = plotfileDirectory + commandLineArguments.saveFilename
		pathTemplate = pathTemplate + '--k_nx={}--k_ny={}--k_nz={}.data'

		writePlotfiles(rundirs, commandLineArguments, pathTemplate)
	else:
		plotter = MPCDAnalysis.InteractivePlotter.Plotter()

		if commandLineArguments.dataDirectory is not None:
			plotfileDirectory = commandLineArguments.dataDirectory

		plotter.provideDataDirectory(theoryDirectory, {'Type': 'Theory'})
		plotter.provideDataDirectory(plotfileDirectory, {'Type': 'Simulation'}, True)
		plotter.setCharacteristicsOrder(['Plotted', 'Type', 'omega_H'])
		plotter.hideCharacteristics(['L_y', 'L_z', 'rho', 't_step'])
		plotter.setSortingCharacteristic('omega_H', True)

		plotter.setYScale('log')
		plotter.setPlotDashedIfNegative(True)

		plotter.run()
