#include <MPCDAnalysis/File.hpp>

#include <OpenMPCD/Exceptions.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>

MPCDAnalysis::File::File(const std::string& filename)
{
	if(boost::filesystem::is_regular_file(filename))
	{
		if(filename.substr(filename.size() - 4) == ".bz2")
		{
			initialize_bz2(filename);
		}
		else
		{
			initialize_regular(filename);
		}
	}
	else if(boost::filesystem::is_regular_file(filename + ".bz2"))
	{
		initialize_bz2(filename + ".bz2");
	}
	else
	{
		OPENMPCD_THROW(OpenMPCD::IOException, "No such file or directory in getInputFilestream.");
	}
}

MPCDAnalysis::File::~File()
{
	for(auto&& stream : streams)
		stream.reset();
}

std::istream& MPCDAnalysis::File::getInputStream()
{
	return *streams.front();
}

bool MPCDAnalysis::File::readLine(std::string& line)
{
	return std::getline(getInputStream(), line) ? true : false;
}

void MPCDAnalysis::File::readNameValuePairs()
{
	std::string line;

	while(getInputStream().peek() == '#')
	{
		readLine(line);
		line = line.substr(1); //skip '#' character

		std::vector<std::string> tokens;
		boost::algorithm::split(tokens, line, boost::is_any_of("#"));

		if(tokens.size() != 2)
			OPENMPCD_THROW(OpenMPCD::IOException, "Unexpected file format.");

		boost::algorithm::trim(tokens[0]);
		boost::algorithm::trim(tokens[1]);

		nameValuePairs[tokens[0]] = tokens[1];
	}
}

bool MPCDAnalysis::File::hasProperty(const std::string& name) const
{
	return nameValuePairs.find(name) != nameValuePairs.end();
}

namespace MPCDAnalysis
{
template<> const double File::getProperty(const std::string& name) const
{
	try
	{
		const auto it = nameValuePairs.find(name);

		if(it == nameValuePairs.end())
			OPENMPCD_THROW(OpenMPCD::InvalidArgumentException, "No such property.");

		return boost::lexical_cast<double>(it->second);
	}
	catch(const boost::bad_lexical_cast&)
	{
		OPENMPCD_THROW(OpenMPCD::IOException, "Failed to cast to requested type.");
	}
}
} //namespace MPCDAnalysis

void MPCDAnalysis::File::initialize_regular(const std::string& filename)
{
	typedef boost::iostreams::filtering_stream<boost::iostreams::input> FilteringStream;

	auto file = std::make_shared<std::ifstream>(filename);
	auto stream = std::make_shared<FilteringStream>();

	stream->push(*file);

	streams.push_back(stream);
	streams.push_back(file);
}

void MPCDAnalysis::File::initialize_bz2(const std::string& filename)
{
	typedef boost::iostreams::filtering_stream<boost::iostreams::input> FilteringStream;

	auto file = std::make_shared<std::ifstream>(filename);
	auto stream = std::make_shared<FilteringStream>();

	stream->push(boost::iostreams::bzip2_decompressor());
	stream->push(*file);

	streams.push_back(stream);
	streams.push_back(file);
}
