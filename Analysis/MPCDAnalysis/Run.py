from .Configuration import Configuration

import enum
import yaml

class Run:
	"""
	Represents a run of OpenMPCD, and its results.
	"""

	class RunState(enum.Enum):
		"""
		Enumerates the state of a run.

		Possible states:
			* `Ready`, if the run is ready for execution or job
			  submission,
			* `Submitted`, if the run has been submitted to the job
			  scheduler,
			* `Running`, if the run is currently being executed,
			* `Completed`, if the run has finished executing.
		"""

		Ready = 1
		Submitted = 2
		Running = 3
		Completed = 4

		def __str__(self):
			return self.name



	def __init__(self, rundir, pathTranslatorsServerToLocal = []):
		"""
		The constructor.

		@param[in] rundir
		           The directory the run is saved in.
		@param[in] pathTranslatorsServerToLocal
		           A list of functions that translate server paths to local
		           ones.
		"""

		self.rundir = rundir

		self.pathTranslatorsServerToLocal = pathTranslatorsServerToLocal

		self.config = None


	def getPath(self):
		"""
		Returns the rundir path.
		"""

		return self.rundir


	def getIdentifier(self):
		"""
		Returns a string that can be used to identify this particular run.

		@throw Exception Throws if `self.getState() != self.RunState.Completed`.
		"""

		if self.getState() != self.RunState.Completed:
			raise Exception("Run not yet completed")

		configHash = self._getSHA256ByFilePath(self.rundir + "/config.txt")
		revision = open(self.rundir + "/git-revision").read()
		metadataHash = self._getSHA256ByFilePath(self.rundir + "/metadata.txt")
		seed = open(self.rundir + "/rngSeed.txt").read()

		assert revision.count("\n") == 0
		assert seed.count("\n") == 0

		return configHash + ":" + revision + ":" + metadataHash + ":" + seed


	def getState(self):
		"""
		Returns the run state as an instance of `RunState`.
		"""

		import glob
		import os.path

		if not os.path.isfile(self.rundir + "/config.txt"):
			raise Exception("No configuration file found")

		if os.path.isfile(self.rundir + "/metadata.txt"):
			return self.RunState.Completed

		if glob.glob(self.rundir + "/input/srun-*.out"):
			return self.RunState.Running

		if self.hasParentRun():
			return self._getParentRun().getState()

		if os.path.exists(self.rundir + "/input/jobid"):
			return self.RunState.Submitted

		if os.path.isfile(self.rundir + "/input/job.slrm"):
			return self.RunState.Ready

		raise Exception(self.rundir + " does not seem to be a rundir")


	def hasParentRun(self):
		"""
		Returns whether this run has been executed by a job script in another
		run.
		"""

		import os.path

		return os.path.isfile(self.rundir + "/input/parent-job-path.txt")


	def getJobBatchSize(self):
		"""
		Returns the number of runs that run in parallel on this job.
		"""

		topRun = self._getParentRun()

		ret = 0
		with open(topRun.getPath() + "/input/job.slrm", "r") as jobfile:
			for line in jobfile:
				if line.startswith("srun"):
					ret += 1

		assert ret > 0

		return ret


	def getConfiguration(self):
		"""
		Returns the configuration instance for this run.
		"""

		if self.config is None:
			self.config = Configuration(self.rundir + "/config.txt")

		return self.config


	def getNumberOfCompletedSweeps(self):
		"""
		Returns the number of sweeps that have been performed after the warmup
		phase, as configured in `mpc.warmupSteps`, or `0` if the run has not
		yet completed.
		"""

		import os.path

		metadataPath = self.rundir + "/metadata.txt"

		if not os.path.isfile(metadataPath):
			return 0

		config = yaml.safe_load(open(metadataPath, "r"))
		return config["numberOfCompletedSweeps"]


	def _getSHA256ByFilePath(self, path):
		"""
		Returns the SHA256 hash of the file at `path`, as a string of
		hexadecimal digits (with lowercase letters).
		"""

		import hashlib

		return hashlib.sha256(open(path, "r").read()).hexdigest()


	def _applyPathTranslatorsServerToLocal(self, s):
		"""
		Applies all path translators in `self.pathTranslatorsServerToLocal` on
		the given string, and returns the result.
		"""

		ret = s
		for translator in self.pathTranslatorsServerToLocal:
			ret = translator(ret)

		return ret


	def _getParentRun(self):
		"""
		For runs that are part of a job that runs multiple executions in
		parallel, return the run that contains the job script; otherwise, return
		this run.
		"""

		if self.hasParentRun():
			with open(self.rundir + "/input/parent-job-path.txt", "r") as f:
				rundir = self._applyPathTranslatorsServerToLocal(f.read())
				parentRun = Run(rundir)
				return parentRun

		return self


