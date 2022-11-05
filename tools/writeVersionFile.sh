#!/bin/bash

getGitCommitString()
{
	local ret=$1

	result='NO REPOSITORY FOUND'
	if git status > /dev/null 2>&1
	then
		result=`git describe --always --dirty=+MODIFICATIONS --abbrev=100`
		untracked=`git ls-files --other --full-name --directory --exclude=builds --exclude-standard :/`
		if [ ! -z "$untracked" ]
		then
			result+='+UNTRACKED'
		fi
	else
		echo "WARNING: could not find current git repository information in $0" 1>&2
		exit 0
	fi

	eval $ret=\$result
}

gitCommitString=
getGitCommitString gitCommitString

MYPATH=$( cd "$(dirname "$0")" ; pwd -P )
TMPFILE=`mktemp`
cp "$MYPATH/../src/OpenMPCD/getGitCommitIdentifier.cpp.template" "$TMPFILE"
sed -i "s/OPENMPCD_VERSION/$gitCommitString/g" "$TMPFILE"
diff "$TMPFILE" "$MYPATH/../src/OpenMPCD/getGitCommitIdentifier.cpp" > /dev/null 2>&1
if (($? > 0)) ; then
	cp "$TMPFILE" "$MYPATH/../src/OpenMPCD/getGitCommitIdentifier.cpp"
fi
