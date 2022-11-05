/**
 * @file
 * Tests `OpenMPCD::getGitCommitIdentifier`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/getGitCommitIdentifier.hpp>

#include <boost/regex.hpp>

#include <cstdio>
#include <wait.h>

static const std::string getLocalGitRevision()
{
	FILE* gitHandle=popen("git rev-parse HEAD 2>/dev/null", "r");
	int gitStatus;
	wait(&gitStatus);

	if(WIFEXITED(gitStatus) && WEXITSTATUS(gitStatus)!=0)
	{
		pclose(gitHandle);
		return "";
	}

	char gitMessage[1024]={0};
	const size_t bytesRead =
		fread(gitMessage, 1, sizeof(gitMessage)-1, gitHandle);
	pclose(gitHandle);

	if(bytesRead==0)
		OPENMPCD_THROW(OpenMPCD::Exception, "Failed to read form git.");

	std::string message(gitMessage);
	if(!message.empty() && message.substr(message.size() - 1) == "\n")
		message = message.substr(0, message.size() - 1);

	return message;
}

SCENARIO(
	"`OpenMPCD::Scalar::getGitCommitIdentifier`",
	"")
{
	const std::string localRevision = getLocalGitRevision();
	REQUIRE(getLocalGitRevision() == localRevision); //consistency

	const std::string commitIdentifier = OpenMPCD::getGitCommitIdentifier();
	REQUIRE(OpenMPCD::getGitCommitIdentifier() == commitIdentifier);
		//consistency

	const std::string expectedResultRegexString =
		"[0-9a-f]{40}"
		"(?:\\+MODIFICATIONS)?"
		"(?:\\+UNTRACKED)?";
	const boost::regex expectedResultRegex(expectedResultRegexString);

	REQUIRE(boost::regex_match(commitIdentifier, expectedResultRegex));

	if(localRevision.empty())
	{
		WARN(
			"Cannot do full test of `OpenMPCD::Scalar::getGitCommitIdentifier`, "
			"because the git revision information could not be retrieved "
			"locally.");
	}
	else
	{
		REQUIRE(OpenMPCD::getGitCommitIdentifier() == getLocalGitRevision());
	}
}
