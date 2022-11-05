#include <OpenMPCD/VTFSnapshotFile.hpp>

#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/Types.hpp>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>

#include <iomanip>      // std::setprecision
#include <limits>
#include <sstream>
#include <iostream>

using namespace OpenMPCD;

VTFSnapshotFile::VTFSnapshotFile(const std::string& path)
  : structureBlockProcessed(false),
    primarySimulationVolumeSizeSet(false),
    primarySimulationVolumeSize(0, 0, 0)
{
	std::ios_base::openmode openmode;

	if(boost::filesystem::is_regular_file(path))
	{
		writeModeFlag = false;
		openmode = std::ios_base::in;
	}
	else
	{
		writeModeFlag = true;
		openmode = std::ios_base::out;

		(boost::filesystem::ofstream(path)); //create file
	}

	file.open(path.c_str(), openmode);

	if(file.fail())
	{
		std::string msg = "Failed to open snapshot file: ";
		msg += path;

		OPENMPCD_THROW(IOException, msg);
	}

	if(isInWriteMode())
	{
		file.precision(std::numeric_limits<FP>::digits10 + 2);
	}
	else
	{
		readStructureBlock();
	}
}

VTFSnapshotFile::~VTFSnapshotFile()
{
	if(isInWriteMode() && !structureBlockProcessed)
		writeStructureBlock();
}

void VTFSnapshotFile::setPrimarySimulationVolumeSize(
	const FP& x, const FP& y, const FP& z)
{
	assertWriteMode();
	assertStructureBlockNotProcessed();

	primarySimulationVolumeSize.setX(x);
	primarySimulationVolumeSize.setY(y);
	primarySimulationVolumeSize.setZ(z);

	primarySimulationVolumeSizeSet = true;
}

const std::pair<std::size_t, std::size_t>
VTFSnapshotFile::declareAtoms(const std::size_t count)
{
	assertWriteMode();
	assertStructureBlockNotProcessed();

	if(count == 0)
		return std::pair<std::size_t, std::size_t>(0, 0);

	AtomRange range;
	range.first = getNumberOfAtoms();
	range.last = range.first + count - 1;
	range.properties = defaultAtomProperties;

	atomRanges.push_back(range);

	return std::make_pair(range.first, range.last);
}

const std::pair<std::size_t, std::size_t>
VTFSnapshotFile::declareAtoms(
	const std::size_t count, const FP radius,
	const std::string& name, const std::string& type)
{
	assertWriteMode();
	assertStructureBlockNotProcessed();

	if(count == 0)
		return std::pair<std::size_t, std::size_t>(0, 0);

	if(radius < 0 && radius != -1)
		OPENMPCD_THROW(InvalidArgumentException, "Invalid radius given");

	if(name.size() > 16)
		OPENMPCD_THROW(InvalidArgumentException, "`name` too long");

	if(type.size() > 16)
		OPENMPCD_THROW(InvalidArgumentException, "`type` too long");

	AtomRange range;

	range.first = getNumberOfAtoms();
	range.last = range.first + count - 1;
	range.properties = defaultAtomProperties;

	if(radius != -1)
		range.properties.radius = radius;

	if(!name.empty())
		range.properties.name = name;

	if(!type.empty())
		range.properties.type = type;

	atomRanges.push_back(range);

	return std::make_pair(range.first, range.last);
}

const VTFSnapshotFile::AtomProperties&
	VTFSnapshotFile::getAtomProperties(const std::size_t atomID) const
{
	typedef std::list<AtomRange>::const_iterator It;
	for(It it = atomRanges.begin(); it != atomRanges.end(); ++it)
	{
		if(atomID > it->last)
			continue;

		return it->properties;
	}

	OPENMPCD_THROW(OutOfBoundsException, "Invalid atom ID given.");
}

void VTFSnapshotFile::declareBond(std::size_t atom1, std::size_t atom2)
{
	assertWriteMode();
	assertStructureBlockNotProcessed();

	if(!isValidAtomID(atom1) || !isValidAtomID(atom2))
		OPENMPCD_THROW(OutOfBoundsException, "Invalid atom ID given.");

	if(atom1 == atom2)
		OPENMPCD_THROW(InvalidArgumentException, "atom1 == atom2");

	if(atom2 < atom1)
		std::swap(atom1, atom2);

	const std::pair<std::size_t, std::size_t> bond(atom1, atom2);

	if(bonds.count(bond))
		OPENMPCD_THROW(InvalidArgumentException, "Bond declared already.");

	bonds.insert(bond);
}

bool VTFSnapshotFile::hasBond(std::size_t atom1, std::size_t atom2) const
{
	if(!isValidAtomID(atom1) || !isValidAtomID(atom2))
		OPENMPCD_THROW(OutOfBoundsException, "Invalid atom ID given.");

	if(atom1 == atom2)
		OPENMPCD_THROW(InvalidArgumentException, "atom1 == atom2");

	if(atom2 < atom1)
		std::swap(atom1, atom2);

	const std::pair<std::size_t, std::size_t> bond(atom1, atom2);

	return bonds.count(bond);
}

void VTFSnapshotFile::writeTimestepBlock(
	const FP* const positions, const FP* const velocities)
{
	assertWriteMode();

	if(!structureBlockProcessed)
		writeStructureBlock();

	if(!positions)
		OPENMPCD_THROW(NULLPointerException, "positions");

	file << "timestep\n";

	for(std::size_t i=0; i<getNumberOfAtoms(); ++i)
	{
		file << positions[3*i + 0] << " ";
		file << positions[3*i + 1] << " ";
		file << positions[3*i + 2];

		if(velocities)
		{
			file << " " << velocities[3*i + 0];
			file << " " << velocities[3*i + 1];
			file << " " << velocities[3*i + 2];
		}

		file << "\n";
	}
}

bool VTFSnapshotFile::readTimestepBlock(
	FP* const positions,
	FP* const velocities,
	bool* const velocitiesEncountered)
{
	assertReadMode();

	const boost::char_separator<char> separator(" \t");
	typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
	typedef Tokenizer::iterator TokenIt;

	{
		std::string line;
		do
		{
			if(file.eof())
				return false;
			line = getLine(true);
		} while(line.empty());

		Tokenizer tokenizer(line, separator);

		TokenIt it = tokenizer.begin();
		const std::string firstToken = *it;
		++it;

		if(	firstToken == "t" || firstToken == "timestep" ||
			firstToken == "c" || firstToken == "coordinates")
		{
			if(it != tokenizer.end())
			{
				const std::string secondToken = *it;
				++it;

				if(it != tokenizer.end())
					OPENMPCD_THROW(MalformedFileException, "Too many tokens.");

				if(secondToken == "i" || secondToken == "indexed")
					OPENMPCD_THROW(
						UnimplementedException,
						"Only ordered timestep blocks are supported.");

				if(secondToken != "o" && secondToken != "ordered")
					OPENMPCD_THROW(MalformedFileException, "Bad second token.");
			}
		}
	}

	if(getNumberOfAtoms() == 0)
		return true;

	bool lineWithVelocitiesEncountered = false;
	bool lineWithoutVelocitiesEncountered = false;
	std::size_t atom = 0;
	while(atom < getNumberOfAtoms())
	{
		if(file.eof())
			break;

		const std::string line = getLine(true);

		if(line.empty())
			continue;

		Tokenizer tokenizer(line, separator);

		unsigned int token = 0;
		for(TokenIt it = tokenizer.begin(); it != tokenizer.end(); ++it)
		{
			if(token < 3 && positions)
				positions[3 * atom + token] = lexicalCast<FP>(*it);

			if(token >= 3 && token < 6 && velocities)
				velocities[3 * atom + token - 3] = lexicalCast<FP>(*it);

			++token;
		}

		if(token == 3)
		{
			lineWithoutVelocitiesEncountered = true;
		}
		else if(token == 6)
		{
			lineWithVelocitiesEncountered = true;
		}
		else
		{
			OPENMPCD_THROW(MalformedFileException, "Bad number of positions.");
		}

		++atom;
	}

	if(velocitiesEncountered)
		*velocitiesEncountered = lineWithVelocitiesEncountered;

	if(atom == 0)
		return false;

	if(lineWithVelocitiesEncountered && lineWithoutVelocitiesEncountered)
	{
		OPENMPCD_THROW(
			MalformedFileException,
			"Some lines contained velocity information, while others did not.");
	}

	if(atom != getNumberOfAtoms())
		OPENMPCD_THROW(MalformedFileException, "Incomplete timestep block.");

	return true;
}

void VTFSnapshotFile::assertWriteMode() const
{
	if(!isInWriteMode())
		OPENMPCD_THROW(
			InvalidCallException,
			"Tried to change snapshot in read mode.");
}

void VTFSnapshotFile::assertReadMode() const
{
	if(!isInReadMode())
		OPENMPCD_THROW(
			InvalidCallException,
			"Instance is not in read mode.");
}

void VTFSnapshotFile::assertStructureBlockNotProcessed() const
{
	if(structureBlockProcessed)
		OPENMPCD_THROW(
			InvalidCallException,
			"The structure block has been processed already.");
}

void VTFSnapshotFile::writeStructureBlock()
{
	assertWriteMode();
	assertStructureBlockNotProcessed();

	if(primarySimulationVolumeSizeIsSet())
	{
		file << "pbc ";
		file << primarySimulationVolumeSize.getX() << " ";
		file << primarySimulationVolumeSize.getY() << " ";
		file << primarySimulationVolumeSize.getZ() << "\n";
	}

	writeAtomLines();

	writeBondLines();

	structureBlockProcessed = true;
}

void VTFSnapshotFile::readStructureBlock()
{
	assertReadMode();
	assertStructureBlockNotProcessed();

	while(file.good())
	{
		const std::string line = getLine(false);

		if(line.empty())
		{
			//empty line; do nothing
		}
		else if(line[0] == '#')
		{
			//comment line; do nothing
		}
		else if(line[0] == 'b')
		{
			readBondLine(line);
		}
		else if(line[0] == 'p' || line[0] == 'u')
		{
			readUnitcellLine(line);
		}
		else if(
			line[0] == 't' || line[0] == 'c' ||
			line[0] == 'i' || line[0] == 'o')
		{
			//timestep line; do not extract it from the file stream,
			//and stop reading structure block
			break;
		}
		else
		{
			readAtomLine(line);
		}

		getLine(true); //consume line from input stream
	}

	structureBlockProcessed = true;
}

void VTFSnapshotFile::readUnitcellLine(const std::string& line)
{
	assertReadMode();
	assertStructureBlockNotProcessed();

	//start of the line contains the line type indicator
	std::string regexString = "(?:(?:p|pbc)|(?:u|unitcell))";

	//then, three numbers follow: a, b, and c
	for(unsigned int i=0; i<3; ++i)
		regexString += "\\s+(\\S+)";

	//then, another three numbers may follow: alpha, beta, and gamma
	regexString += "(?:";
	for(unsigned int i=0; i<3; ++i)
		regexString += "\\s+(\\S+)";
	regexString += ")?";

	//end of string may contain whitespace
	regexString+= "\\s*";

	const boost::regex re(regexString);
	boost::cmatch captures;

	if(!boost::regex_match(line.c_str(), captures, re))
		OPENMPCD_THROW(MalformedFileException, "Malformed line:\n" + line);

	OPENMPCD_DEBUG_ASSERT(captures.size() == 1 + 6);
	OPENMPCD_DEBUG_ASSERT(captures[1].matched);
	OPENMPCD_DEBUG_ASSERT(captures[2].matched);
	OPENMPCD_DEBUG_ASSERT(captures[3].matched);

	primarySimulationVolumeSize.setX(lexicalCast<FP>(captures.str(1)));
	primarySimulationVolumeSize.setY(lexicalCast<FP>(captures.str(2)));
	primarySimulationVolumeSize.setZ(lexicalCast<FP>(captures.str(3)));

	primarySimulationVolumeSizeSet = true;
}

void VTFSnapshotFile::writeAtomLines()
{
	assertWriteMode();
	assertStructureBlockNotProcessed();

	typedef std::list<AtomRange>::const_iterator It;
	for(It it = atomRanges.begin(); it != atomRanges.end(); ++it)
	{
		file << "atom " << it->first;
		if(it->first != it->last)
			file << ":" << it->last;

		if(it->properties.name.is_initialized())
			file << " name " << it->properties.name.get();

		if(it->properties.type.is_initialized())
			file << " type " << it->properties.type.get();

		if(it->properties.radius.is_initialized())
			file << " radius " << it->properties.radius.get();

		file << "\n";
	}
}

void VTFSnapshotFile::readAtomLine(const std::string& line_)
{
	assertReadMode();
	assertStructureBlockNotProcessed();

	const boost::regex commaRegex("\\s*,\\s*");
	const std::string line = boost::regex_replace(line_, commaRegex, ", ");

	const boost::char_separator<char> separator(" \t");
	typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
	Tokenizer tokenizer(line, separator);

	const boost::regex aidRegex("([0-9]+)(?::([0-9]+))?(,)?");
	const boost::regex aidDefaultRegex("default(,)?");

	AtomProperties properties = defaultAtomProperties;
	std::vector<std::pair<std::size_t, std::size_t> > newRanges;
	std::string optionName;
	bool isNewDefault = false;
	bool expectAID = true;
	for(Tokenizer::iterator it = tokenizer.begin(); it != tokenizer.end(); ++it)
	{
		if(it == tokenizer.begin() && (*it == "a" || *it == "atom"))
			continue;

		using boost::regex_match;

		boost::cmatch capturesDefault;
		if(expectAID &&
			regex_match(it->c_str(), capturesDefault, aidDefaultRegex))
		{
			OPENMPCD_DEBUG_ASSERT(capturesDefault.size() == 1 + 1);

			if(!capturesDefault[1].matched)
				expectAID = false;

			isNewDefault = true;

			continue;
		}

		boost::cmatch captures;
		if(expectAID && regex_match(it->c_str(), captures, aidRegex))
		{
			OPENMPCD_DEBUG_ASSERT(captures.size() == 1 + 3);
			OPENMPCD_DEBUG_ASSERT(captures[1].matched);

			const std::size_t first = lexicalCast<std::size_t>(captures.str(1));
			const std::size_t last =
				captures[2].matched ?
					lexicalCast<std::size_t>(captures.str(2)) : first;

			if(last < first)
				OPENMPCD_THROW(
					MalformedFileException, "Malformed line:\n" + line);

			newRanges.push_back(std::make_pair(first, last));

			if(!captures[3].matched)
				expectAID = false;

			continue;
		}

		if(expectAID)
			OPENMPCD_THROW(MalformedFileException, "Malformed line:\n" + line);

		//next is either an option name or an option value
		if(optionName.empty())
		{
			//expect option name next
			if(*it == "n" || *it == "name")
			{
				optionName = "name";
			}
			else if(*it == "t" || *it == "type")
			{
				optionName = "type";
			}
			else if(*it == "r" || *it == "radius")
			{
				optionName = "radius";
			}
			else
			{
				OPENMPCD_THROW(
					MalformedFileException, "Unknown option: " + *it);
			}
		}
		else
		{
			//expect option value
			if(optionName == "name")
			{
				if(it->size() > 16)
					OPENMPCD_THROW(
						MalformedFileException, "Name too long: " + *it);

				properties.name = *it;
			}
			else if(optionName == "type")
			{
				if(it->size() > 16)
					OPENMPCD_THROW(
						MalformedFileException, "Type too long: " + *it);

				properties.type = *it;
			}
			else if(optionName == "radius")
			{
				properties.radius = lexicalCast<FP>(*it);

				if(properties.radius.get() < 0)
					OPENMPCD_THROW(MalformedFileException, "Bad radius: " + *it);
			}
			else
			{
				OPENMPCD_THROW(
					UnimplementedException, "Unknown option: " + *it);
			}

			optionName.clear();
		}
	}

	if(!optionName.empty())
		OPENMPCD_THROW(MalformedFileException, "Malformed atom line:\n" + line);

	if(isNewDefault)
		defaultAtomProperties = properties;

	for(std::size_t i=0; i<newRanges.size(); ++i)
	{
		AtomRange newRange;
		newRange.first = newRanges[i].first;
		newRange.last = newRanges[i].second;
		newRange.properties = properties;
		setAtomRange(newRange);
	}
}

void VTFSnapshotFile::setAtomRange(const AtomRange& range)
{
	assertAtomRangesContiguous();
	OPENMPCD_DEBUG_ASSERT(range.first <= range.last);

	if(atomRanges.empty())
	{
		if(range.first != 0)
		{
			AtomRange newRange;
			newRange.first = 0;
			newRange.last = range.first - 1;
			newRange.properties = defaultAtomProperties;

			atomRanges.push_back(newRange);
		}

		atomRanges.push_back(range);
	}

	typedef std::list<AtomRange>::iterator It;
	for(It it = atomRanges.begin(); it != atomRanges.end(); ++it)
	{
		if(range.first > it->last)
			continue;

		OPENMPCD_DEBUG_ASSERT(range.first >= it->first);

		if(range.first > it->first)
		{
			AtomRange prefix = *it;
			prefix.last = range.first - 1;
			atomRanges.insert(it, prefix);

			it->first = range.first;
		}

		OPENMPCD_DEBUG_ASSERT(range.first == it->first);

		while(range.last >= it->last)
		{
			it = atomRanges.erase(it);

			if(it == atomRanges.end())
			{
				atomRanges.push_back(range);
				return;
			}
		}

		OPENMPCD_DEBUG_ASSERT(range.first <= it->first);
		OPENMPCD_DEBUG_ASSERT(range.last >= it->first);
		OPENMPCD_DEBUG_ASSERT(range.last < it->last);

		it->first = range.last + 1;

		OPENMPCD_DEBUG_ASSERT(range.last < it->last);

		atomRanges.insert(it, range);
		return;
	}

	const std::size_t lastAtomID = getNumberOfAtoms() - 1;
	if(lastAtomID + 1 < range.first)
	{
		AtomRange newRange;
		newRange.first = lastAtomID + 1;
		newRange.last = range.first - 1;
		newRange.properties = defaultAtomProperties;

		atomRanges.push_back(newRange);
	}

	atomRanges.push_back(range);
}

void VTFSnapshotFile::assertAtomRangesContiguous() const
{
	std::size_t nextFirst = 0;
	typedef std::list<AtomRange>::const_iterator It;
	for(It it = atomRanges.begin(); it != atomRanges.end(); ++it)
	{
		if(it->first != nextFirst)
			OPENMPCD_THROW(Exception, "Atom ranges not contiguous");

		if(it->first > it->last)
			OPENMPCD_THROW(Exception, "Atom range inverted");

		nextFirst = it->last + 1;
	}
}

void VTFSnapshotFile::writeBondLines()
{
	assertWriteMode();
	assertStructureBlockNotProcessed();

	typedef std::set<std::pair<std::size_t, std::size_t> > Set;
	for(Set::const_iterator it = bonds.begin(); it != bonds.end(); ++it)
		file << "bond " << it->first << ":" << it->second << "\n";
}

void VTFSnapshotFile::readBondLine(const std::string& line_)
{
	assertReadMode();
	assertStructureBlockNotProcessed();

	const boost::regex commaRegex("\\s*,\\s*");
	const std::string line = boost::regex_replace(line_, commaRegex, ", ");

	const boost::char_separator<char> separator(" \t");
	typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
	Tokenizer tokenizer(line, separator);

	const boost::regex bondSpecifierRegex("([0-9]+)(::?)([0-9]+)(,)?");

	bool expectBondSpecifier = true;
	for(Tokenizer::iterator it = tokenizer.begin(); it != tokenizer.end(); ++it)
	{
		if(it == tokenizer.begin() && (*it == "b" || *it == "bond"))
			continue;

		if(!expectBondSpecifier)
			OPENMPCD_THROW(MalformedFileException, "Malformed line:\n" + line);

		using boost::regex_match;

		boost::cmatch captures;
		if(!regex_match(it->c_str(), captures, bondSpecifierRegex))
			OPENMPCD_THROW(MalformedFileException, "Malformed line:\n" + line);

		OPENMPCD_DEBUG_ASSERT(captures.size() == 1 + 4);
		OPENMPCD_DEBUG_ASSERT(captures[1].matched);
		OPENMPCD_DEBUG_ASSERT(captures[2].matched);
		OPENMPCD_DEBUG_ASSERT(captures[3].matched);

		const std::size_t first = lexicalCast<std::size_t>(captures.str(1));
		const std::size_t last =
			captures[3].matched ?
				lexicalCast<std::size_t>(captures.str(3)) : first;

		if(last <= first)
			OPENMPCD_THROW(
				MalformedFileException, "Malformed line:\n" + line);

		if(!isValidAtomID(first) || !isValidAtomID(last))
			OPENMPCD_THROW(
				MalformedFileException, "Malformed line:\n" + line);


		OPENMPCD_DEBUG_ASSERT(
			captures.str(2) == ":" || captures.str(2) == "::");

		const bool isBondChain = captures.str(2).length() == 2;


		if(isBondChain)
		{
			for(std::size_t current = first; current < last; ++current)
			{
				const std::pair<std::size_t, std::size_t>
					bond(current, current + 1);

				if(bonds.count(bond))
				{
					std::stringstream ss;
					ss << "Bond declared twice: ";
					ss << first << ":" << last << "\n";
					ss << "In line: " << line;
					OPENMPCD_THROW(MalformedFileException, ss.str());
				}

				bonds.insert(bond);
			}
		}
		else
		{
			const std::pair<std::size_t, std::size_t> bond(first, last);

			if(bonds.count(bond))
			{
				std::stringstream ss;
				ss << "Bond declared twice: ";
				ss << first << ":" << last << "\n";
				ss << "In line: " << line;
				OPENMPCD_THROW(MalformedFileException, ss.str());
			}

			bonds.insert(bond);
		}


		if(!captures[4].matched)
			expectBondSpecifier = false;
	}
}

const std::string VTFSnapshotFile::getLine(const bool extract)
{
	assertReadMode();

	if(!file.good())
		return "";

	if(file.peek() == std::char_traits<std::fstream::char_type>::eof())
	{
		//necessary workaround, since `std::getline` would set `failbit`
		//otherwise
		file.clear();
		file.setstate(std::ios_base::eofbit);
		return "";
	}

	std::string line;

	if(extract)
	{
		std::getline(file, line);
		return stripLeadingWhitespace(line);
	}

	const std::fstream::pos_type position = file.tellg();
	if(position == std::fstream::pos_type(-1))
		OPENMPCD_THROW(IOException, "Failed to read line.");

	std::getline(file, line);

	if(file.fail())
		OPENMPCD_THROW(IOException, "Failed to read line.");

	file.clear();
	file.seekg(position);

	if(file.fail())
		OPENMPCD_THROW(IOException, "Failed to read line.");

	return stripLeadingWhitespace(line);
}

const std::string VTFSnapshotFile::stripLeadingWhitespace(
	const std::string& str)
{
	const std::string::size_type firstNonWhitespace =
		str.find_first_not_of(" \t\n");

	if(firstNonWhitespace == std::string::npos)
		return "";

	return str.substr(firstNonWhitespace);
}
