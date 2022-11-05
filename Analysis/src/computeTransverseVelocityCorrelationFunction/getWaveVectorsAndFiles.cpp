#include <computeTransverseVelocityCorrelationFunction/functions.hpp>

#include <MPCDAnalysis/File.hpp>
#include <MPCDAnalysis/Utilities.hpp>

#include <OpenMPCD/Exceptions.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>

const std::map<OpenMPCD::Vector3D<double>, std::vector<std::string>>
	getWaveVectorsAndPaths(const std::vector<std::string>& rundirs)
{
	std::map<OpenMPCD::Vector3D<double>, std::vector<std::string>> ret;

	const std::string filenameRegex = "[0-9]+\\.data.*";

	for(const auto rundir : rundirs)
	{
		const std::vector<std::string> filenames =
			MPCDAnalysis::Utilities::getFilesByRegex(rundir + "/fourierTransformedVelocities", filenameRegex, true);

		for(const auto filename : filenames)
		{
			MPCDAnalysis::File file(filename);

			std::string line;
			file.readLine(line);

			if(line.substr(0, 5) != "#k_n:")
				OPENMPCD_THROW(OpenMPCD::IOException, "Invalid file format.");

			line = line.substr(5);
			boost::algorithm::trim(line);

			std::vector<std::string> tokens;
			boost::algorithm::split(tokens, line, boost::is_any_of(" \t"));

			if(tokens.size() != 3)
				OPENMPCD_THROW(OpenMPCD::IOException, "Invalid file format.");

			boost::algorithm::trim(tokens[0]);
			boost::algorithm::trim(tokens[1]);
			boost::algorithm::trim(tokens[2]);

			const double k_x = boost::lexical_cast<double>(tokens[0]);
			const double k_y = boost::lexical_cast<double>(tokens[1]);
			const double k_z = boost::lexical_cast<double>(tokens[2]);

			const OpenMPCD::Vector3D<double> k(k_x, k_y, k_z);

			ret[k].push_back(filename);
		}
	}

	return ret;
}
