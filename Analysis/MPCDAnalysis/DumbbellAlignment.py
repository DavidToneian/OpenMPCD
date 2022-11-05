from .EmpiricalDistribution import EmpiricalDistribution

import errno

class DumbbellAlignment:
	def __init__(self, rundir):
		try:
			self.xx = EmpiricalDistribution(rundir + "/dumbbellBondXXHistogram.data")
			self.yy = EmpiricalDistribution(rundir + "/dumbbellBondYYHistogram.data")
			self.xy = EmpiricalDistribution(rundir + "/dumbbellBondXYHistogram.data")
		except (IOError, OSError) as e:
			if e.errno != errno.ENOENT:
				raise
			self.xx = EmpiricalDistribution(rundir + "/dumbbellBondXXHistogram.gnuplot")
			self.yy = EmpiricalDistribution(rundir + "/dumbbellBondYYHistogram.gnuplot")
			self.xy = EmpiricalDistribution(rundir + "/dumbbellBondXYHistogram.gnuplot")

	def getTan2Chi(self):
		#See Kowalik and Winkler, J. Chem. Phys. 138, 104903 (2013), section V.2
		return 2.0 * self.xy.getSampleMean() / (self.xx.getSampleMean() - self.yy.getSampleMean())
