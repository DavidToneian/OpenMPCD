/**
 * @file
 * Declares the `OpenMPCD::getGitCommitIdentifier` function.
 */

#ifndef OPENMPCD_GETGITCOMMITIDENTIFIER_HPP
#define OPENMPCD_GETGITCOMMITIDENTIFIER_HPP

#include <string>

namespace OpenMPCD
{
	/**
	 * Returns a string identifying the current git commit of the source code.
	 *
	 * The returned string constist of the current git commit hash,
	 * plus the string `+MODIFICATIONS` if tracked files have been modified,
	 * but not yet commited,
	 * plus the string `+UNTRACKED` if there are any files that are neither
	 * tracked nor ignored.
	 *
	 * The returned string is automatically updated if `make` is run from the
	 * git repository, or if any of the `copy` scripts in the `tools` directory
	 * is run from the git repository.
	 *
	 * Returns an empty string if the `git` command cannot be executed,
	 * or if no commit hash could be found (e.g. because there is no git
	 * repository).
	 *
	 * @throw OpenMPCD::Exception Throws if the output of the `git` command
	 *                           could not be read.
	 */
	const std::string getGitCommitIdentifier();
}

#endif
