#include <OpenMPCD/FlowProfile.hpp>

#include <OpenMPCD/AnalyticQuantities.hpp>
#include <OpenMPCD/Exceptions.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <fstream>
#include <limits>

template<typename T>
OpenMPCD::FlowProfile<T>::FlowProfile(
	const unsigned int mpcBoxSizeX,
	const unsigned int mpcBoxSizeY,
	const unsigned int mpcBoxSizeZ,
	const Configuration::Setting& settings)
	: simulationBoxSizeX(mpcBoxSizeX),
	  simulationBoxSizeY(mpcBoxSizeY),
	  simulationBoxSizeZ(mpcBoxSizeZ),
	  cellSubdivisionsX(1), cellSubdivisionsY(1), cellSubdivisionsZ(1),
	  sweepCountPerOutput(0), currentBlockSweepCount(0)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		mpcBoxSizeX != 0, InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		mpcBoxSizeY != 0, InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		mpcBoxSizeZ != 0, InvalidArgumentException);

	if(settings.has("cellSubdivision"))
	{
		settings.read("cellSubdivision.x", &cellSubdivisionsX);
		settings.read("cellSubdivision.y", &cellSubdivisionsY);
		settings.read("cellSubdivision.z", &cellSubdivisionsZ);
	}

	if(settings.has("sweepCountPerOutput"))
		settings.read("sweepCountPerOutput", &sweepCountPerOutput);


	if(cellSubdivisionsX == 0)
		OPENMPCD_THROW(InvalidConfigurationException, "`cellSubdivision.x`");
	if(cellSubdivisionsY == 0)
		OPENMPCD_THROW(InvalidConfigurationException, "`cellSubdivision.y`");
	if(cellSubdivisionsZ == 0)
		OPENMPCD_THROW(InvalidConfigurationException, "`cellSubdivision.z`");
}

template<typename T>
void OpenMPCD::FlowProfile<T>::newSweep()
{
	if(outputBlocks.empty())
	{
		createOutputBlock();
		return;
	}


	++currentBlockSweepCount;

	if(sweepCountPerOutput == 0)
		return;

	if(currentBlockSweepCount != sweepCountPerOutput)
	{
		OPENMPCD_DEBUG_ASSERT(currentBlockSweepCount < sweepCountPerOutput);
		return;
	}

	currentBlockSweepCount = 0;
	createOutputBlock();
}

template<typename T>
void OpenMPCD::FlowProfile<T>::saveToFile(const std::string& path) const
{
	std::ofstream file(path.c_str(), std::ios::trunc);
	file.precision(std::numeric_limits<FP>::digits10 + 2);

	if(!file.good())
		OPENMPCD_THROW(IOException, "Failed to write to file in FlowProfile::saveToFile");

	file << "#x\ty\tz\t";
	file << "meanVelX\tmeanVelY\tmeanVelZ\t";
	file << "stdDevVelX\tstdDevVelY\tstdDevVelZ\t";
	file << "sampleSize";

	for(std::size_t i = 0; i < outputBlocks.size(); ++i)
	{
		file << "\n";
		printOutputBlockToStream(*(outputBlocks[i]), file);
	}
}

template<typename T>
void OpenMPCD::FlowProfile<T>::createOutputBlock()
{
	typedef boost::multi_array<OnTheFlyStatistics<T>, 4> MA;
	typedef boost::shared_ptr<MA> SPMA;

	outputBlocks.push_back(
		SPMA(
			new MA(
				boost::extents
					[simulationBoxSizeX * cellSubdivisionsX]
					[simulationBoxSizeY * cellSubdivisionsY]
					[simulationBoxSizeZ * cellSubdivisionsZ]
					[3]
				)
			)
		);
}

template<typename T>
void OpenMPCD::FlowProfile<T>::printOutputBlockToStream(
	const boost::multi_array<OpenMPCD::OnTheFlyStatistics<T>, 4>& outputBlock,
	std::ostream& stream) const
{
	for(std::size_t x = 0; x < outputBlock.shape()[0]; ++x)
	{
		for(std::size_t y = 0; y < outputBlock.shape()[1]; ++y)
		{
			for(std::size_t z = 0; z < outputBlock.shape()[2]; ++z)
			{
				//If T were an integral type, the division below would not make
				//much sense. Should the need arise, this can be reworked.
				BOOST_STATIC_ASSERT(!boost::is_integral<T>::value);

				stream << x / double(cellSubdivisionsX) << "\t";
				stream << y / double(cellSubdivisionsY) << "\t";
				stream << z / double(cellSubdivisionsZ) << "\t";

				stream <<
					outputBlock[x][y][z][0].getSampleMean() << "\t";
				stream <<
					outputBlock[x][y][z][1].getSampleMean() << "\t";
				stream <<
					outputBlock[x][y][z][2].getSampleMean() << "\t";

				stream <<
					outputBlock[x][y][z][0].
						getSampleStandardDeviation() << "\t";
				stream <<
					outputBlock[x][y][z][1].
						getSampleStandardDeviation() << "\t";
				stream <<
					outputBlock[x][y][z][2].
						getSampleStandardDeviation() << "\t";

				stream << outputBlock[x][y][z][0].getSampleSize() << "\n";

				OPENMPCD_DEBUG_ASSERT(
					outputBlock[x][y][z][0].getSampleSize()
					==
					outputBlock[x][y][z][1].getSampleSize());
				OPENMPCD_DEBUG_ASSERT(
					outputBlock[x][y][z][0].getSampleSize()
					==
					outputBlock[x][y][z][2].getSampleSize());
			}
		}
	}
}

template class OpenMPCD::FlowProfile<float>;
template class OpenMPCD::FlowProfile<double>;
