#include <OpenMPCD/DensityProfile.hpp>

#include <OpenMPCD/AnalyticQuantities.hpp>
#include <OpenMPCD/Exceptions.hpp>

#include <fstream>
#include <limits>

OpenMPCD::DensityProfile::DensityProfile(
	const unsigned int mpcBoxSizeX,
	const unsigned int mpcBoxSizeY,
	const unsigned int mpcBoxSizeZ,
	const Configuration::Setting& settings)
	: fillCount(0),
	  cellSubdivisionsX(1), cellSubdivisionsY(1), cellSubdivisionsZ(1)
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

	if(cellSubdivisionsX == 0)
		OPENMPCD_THROW(InvalidConfigurationException, "`cellSubdivision.x`");
	if(cellSubdivisionsY == 0)
		OPENMPCD_THROW(InvalidConfigurationException, "`cellSubdivision.y`");
	if(cellSubdivisionsZ == 0)
		OPENMPCD_THROW(InvalidConfigurationException, "`cellSubdivision.z`");

	const unsigned int sizeX = mpcBoxSizeX * cellSubdivisionsX;
	const unsigned int sizeY = mpcBoxSizeY * cellSubdivisionsY;
	const unsigned int sizeZ = mpcBoxSizeZ * cellSubdivisionsZ;

	const std::vector<FP> zeroes(sizeZ, 0);

	points.resize(sizeX, std::vector<std::vector<FP> >(sizeY, zeroes));
}

void OpenMPCD::DensityProfile::saveToFile(const std::string& path) const
{
	std::ofstream file(path.c_str(), std::ios::trunc);
	file.precision(std::numeric_limits<FP>::digits10 + 2);

	if(!file.good())
		OPENMPCD_THROW(IOException, "Failed to write to file.");

	for(std::vector<std::vector<std::vector<FP> > >::size_type x=0; x<points.size(); ++x)
	{
		const std::vector<std::vector<FP> >& tmp1 = points[x];
		for(std::vector<std::vector<FP> >::size_type y=0; y<tmp1.size(); ++y)
		{
			const std::vector<FP>& tmp2 = tmp1[y];
			for(std::vector<FP>::size_type z=0; z<tmp2.size(); ++z)
			{
				const FP totalMass = tmp2[z];
				file<<FP(x)/cellSubdivisionsX<<"\t"<<FP(y)/cellSubdivisionsY<<"\t"<<FP(z)/cellSubdivisionsZ<<"\t";
				file<<totalMass / fillCount<<"\n";
			}
		}
	}
}
