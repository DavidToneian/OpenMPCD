#include <OpenMPCD/Histogram.hpp>

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/FilesystemUtilities.hpp>

#include <fstream>
#include <limits>

OpenMPCD::Histogram::Histogram(const std::string& name, const Configuration& conf)
	: underflows(0), overflows(0)
{
	const unsigned int binCount = conf.read<unsigned int>("instrumentation."+name+".binCount");
	bins.resize(binCount, 0);

	conf.read("instrumentation."+name+".low",  &lowEnd);
	conf.read("instrumentation."+name+".high", &highEnd);

	binSize = (highEnd-lowEnd)/binCount;
}

void OpenMPCD::Histogram::fill(const FP val)
{
	if(val<lowEnd)
	{
		++underflows;
		return;
	}

	if(val>=highEnd)
	{
		++overflows;
		return;
	}

	const unsigned int binID=static_cast<unsigned int>((val-lowEnd)/binSize);

	#ifdef OPENMPCD_DEBUG
		if(binID>=bins.size())
			OPENMPCD_THROW(OutOfBoundsException, "Unexpected");
	#endif

	++bins[binID];
}

OpenMPCD::FP OpenMPCD::Histogram::getIntegral() const
{
	FP sum=0;

	for(Container::size_type i=0; i<bins.size(); ++i)
		sum+=bins[i];

	return sum*binSize;
}

const OpenMPCD::Graph OpenMPCD::Histogram::getNormalizedGraph(const FP binPoint) const
{
	if(binPoint<0 || binPoint>1)
		OPENMPCD_THROW(InvalidArgumentException, "binPoint");

	Graph ret;

	const FP integral=getIntegral();

	for(Container::size_type i=0; i<bins.size(); ++i)
	{
		const FP binPosition=lowEnd+i*binSize+binPoint*binSize;
		const FP binValue=bins[i]/integral;

		ret.addPoint(binPosition, binValue);
	}

	return ret;
}

void OpenMPCD::Histogram::save(const std::string& filename, const FP binPoint) const
{
	FilesystemUtilities::ensureParentDirectory(filename);

	std::ofstream file(filename.c_str(), std::ios::trunc);
	file.precision(std::numeric_limits<FP>::digits10 + 2);

	file<<"#underflows = "<<underflows<<"\n";
	file<<"#overflows ="  <<overflows <<"\n";

	for(Container::size_type i=0; i<bins.size(); ++i)
	{
		const FP binPosition=lowEnd+i*binSize+binPoint*binSize;
		const FP binValue=bins[i];

		file<<binPosition<<"\t"<<binValue<<"\n";
	}
}
