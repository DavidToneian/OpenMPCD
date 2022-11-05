import os.path
import pygit2
import yaml

class ProgramVersionDatabase:
	"""
	Provides information on the versions of the OpenMPCD program.
	"""

	def __init__(self):
		"""
		The constructor.

		This will require a file `.OpenMPCD/config/ProgramVersionDatabase.yaml`
		to be readable, and contain in `repositoryPath` the path to the git
		repository of the `OpenMPCD` project. The path may contain an initial '~'
		character, which will be expanded to the user's home directory.
		"""

		configPath = \
			os.path.expanduser("~/.OpenMPCD/config/ProgramVersionDatabase.yaml")

		config = yaml.safe_load(open(configPath, "r"))

		repoPath = os.path.expanduser(config["repositoryPath"])

		self.repository = pygit2.Repository(repoPath)

		self.knownBugs = {
			1: "ee7ec551c6f6d038d2a1195f35e4da8264f7f8e5",
			8: "62e287cb9d60906bf4b514291f42fc882d2d4f90",
			15: "9daa26741fb622759b4b32bef542441b693c7e22",
			18: "28b668cdf97e4a478c31bd17702bfacfddd2ed07",
			19: "53af091fd877cb52dcbe150aeaa5a8e03ebb67ca",
			}


	def returnCurrentRepositoryDescription(self):
		"""
		Returns the current commit SHA1 id, with the string "+MODIFICATIONS"
		appended if any of the currently tracked files have been modified, and
		with the string "+UNTRACKED" appended if there are files which are not
		tracked.
		"""

		modificationsString = "+MODIFICATIONS"
		untrackedString = "+UNTRACKED"

		description = \
			self.repository.describe(
				describe_strategy = pygit2.GIT_DESCRIBE_ALL,
				abbreviated_size = 100,
				dirty_suffix = modificationsString)

		modified = False


		if description.endswith(modificationsString):
			modified = True
			description = description[:-len(modificationsString)]

		commit = str(self.repository.revparse_single(description).id)

		ret = commit
		if modified:
			ret += modificationsString

		for path, flags in self.repository.status().items():
			if flags & pygit2.GIT_STATUS_WT_NEW:
				if not self.repository.path_is_ignored(path):
					ret += untrackedString
					break

		return ret


	def commitContainsNoKnownBugs(self, commit):
		"""
		Returns whether the specified `commit` contains no known bugs.
		"""

		for bugID, fixingCommit in self.knownBugs.items():
			if fixingCommit is None:
				return False

			if not self._gitCommitAContainsCommitB(commit, fixingCommit):
				return False

		return True


	def currentCommitContainsNoKnownBugs(self):
		"""
		Returns whether the current commit contains no known bugs.

		Untracked and unstaged changes are ignored for this function.
		"""

		return self.commitContainsNoKnownBugs(self._getCurrentGitCommit())


	def _getCurrentGitCommit(self):
		"""
		Returns the commit ID of the current commit.

		Untracked and unstaged changes have no influence on the returned value.
		"""

		description = \
			self.repository.describe(
				describe_strategy = pygit2.GIT_DESCRIBE_ALL,
				abbreviated_size = 100)

		return self.repository.revparse_single(description).id


	def _gitCommitAContainsCommitB(self, commitA, commitB):
		"""
		Returns whether the commit specified in `commitA` contains in its
		history the commit specified in `commitB`.
		"""

		if not isinstance(commitB, pygit2.Oid):
			commitB = pygit2.Oid(hex = commitB)

		for commit in self.repository.walk(commitA):
			if commit.id == commitB:
				return True

		return False


	def _currentCommitContainsCommitB(self, commitB):
		"""
		Returns whether the current commit conatins in its history `commitB`.
		"""

		return \
			self._gitCommitAContainsCommitB(
				self._getCurrentGitCommit(), commitB)
