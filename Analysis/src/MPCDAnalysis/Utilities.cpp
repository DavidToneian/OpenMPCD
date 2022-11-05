#include <MPCDAnalysis/Utilities.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/regex.hpp>

const std::vector<std::string> MPCDAnalysis::Utilities::getFilesByRegex(
	const std::string& dirPath, const std::string& filenameRegex, const bool prependDirPath)
{
	std::vector<std::string> ret;

	if(!boost::filesystem::is_directory(dirPath))
		return ret;

	boost::regex regex(filenameRegex);

	for(auto&& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dirPath), {}))
	{
		if(!boost::filesystem::is_regular_file(entry))
			continue;

		std::string filename = entry.path().filename().string();

		if(!boost::regex_match(filename, regex))
			continue;

		if(prependDirPath)
			filename = dirPath + "/" + filename;

		ret.push_back(filename);
	}

	return ret;
}
