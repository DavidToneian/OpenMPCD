#include <computeTransverseVelocityCorrelationFunction/functions.hpp>

#include <MPCDAnalysis/File.hpp>
#include <MPCDAnalysis/Utilities.hpp>

#include <OpenMPCD/OnTheFlyStatisticsDDDA.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>

typedef OpenMPCD::Vector3D<double> RealVector;
typedef OpenMPCD::Vector3D<std::complex<double>> ComplexVector;
typedef
	std::vector<
		std::pair<double, OpenMPCD::OnTheFlyStatisticsDDDA<double>>> TVACFFT;
			//transverse velocity autocorrelation function, Fourier-transformed


static const std::vector<std::pair<double, ComplexVector>>
	getDataFromFile(
		const std::string& path, const ComplexVector& k,
		const double minimumTime,
		std::ostream* const ostream)
{
	if(ostream)
		*ostream << "Reading file " << path << "\n";

	std::vector<std::pair<double, ComplexVector>> data;
	data.reserve(10000);
	std::string line;

	MPCDAnalysis::File file(path);
	file.readLine(line); //discard first comment line describing the k-vector

	while(file.readLine(line))
	{
		std::vector<std::string> tokens;
		boost::algorithm::split(tokens, line, boost::is_any_of(" \t"));

		if(tokens.size() != 7)
			OPENMPCD_THROW(OpenMPCD::IOException, "Invalid file format.");

		double components[7];
		for(unsigned int i=0; i<7; ++i)
		{
			boost::algorithm::trim(tokens[i]);
			components[i] = boost::lexical_cast<double>(tokens[i]);
		}

		const double t = components[0];

		if(t < minimumTime)
			continue;

		const std::complex<double> vx(components[1], components[2]);
		const std::complex<double> vy(components[3], components[4]);
		const std::complex<double> vz(components[5], components[6]);
		const ComplexVector v(vx, vy, vz);

		const ComplexVector vT = v.getPerpendicularTo(k);

		data.push_back(std::make_pair(t, vT));
	}

	return data;
}

static const TVACFFT getTVACFFT(
	const std::vector<std::string>& dataPaths,
	const ComplexVector& k, const double kT,
	const double minimumTime,
	const double timestep,
	const unsigned int maxStride,
	std::size_t* const sampleSize,
	std::ostream* const ostream)
{
	typedef std::vector<std::pair<double, ComplexVector>> MeasurementSet;

	TVACFFT tvacfft;

	std::vector<MeasurementSet> measurements;
	measurements.reserve(dataPaths.size());

	for(const auto& filePath : dataPaths)
	{
		measurements.push_back(
			getDataFromFile(filePath, k, minimumTime, ostream));
	}

	std::size_t totalMeasurementCount = 0;
	for(const auto& measurementSet : measurements)
		totalMeasurementCount += measurementSet.size();

	if(sampleSize)
		*sampleSize = totalMeasurementCount;

	if(ostream)
	{
		*ostream << "A total of " << totalMeasurementCount
		         << " measurement points have been read.\n";

		*ostream << "Processing strides:";
	}

	for(unsigned int stride = 0; stride < maxStride; ++stride)
	{
		if(stride % 100 == 0 && ostream)
			*ostream << " " << stride << std::flush;

		OpenMPCD::OnTheFlyStatisticsDDDA<double> ddda;

		for(const auto& measurementSet : measurements)
		{
			const std::size_t setSize = measurementSet.size();

			for(std::size_t i=0; i<setSize; ++i)
			{
				if(i + stride >= setSize)
					break;

				const double datum =
					measurementSet[i + stride].second.dot(
						measurementSet[i].second)
					.real();

				ddda.addDatum(datum);
			}
		}

		const double t = stride * timestep;

		tvacfft.push_back(std::make_pair(t, ddda));
	}

	if(ostream)
	 *ostream << "\n";

	return tvacfft;
}

static void writePlotfile(
	const std::vector<std::string>& dataPaths, const std::string& outputPath,
	const RealVector& k_n, const ComplexVector& k, const double kT,
	const double minimumTime,
	const double timestep,
	const unsigned int maxStride,
	const std::vector<std::string>& rundirs,
	const double L_x, const double L_y, const double L_z,
	std::ostream* const ostream,
	std::size_t* const analyzedStepCount)
{
	std::size_t sampleSize;
	const TVACFFT tvacfft =
		getTVACFFT(
			dataPaths, k, kT,
			minimumTime, timestep,
			maxStride, &sampleSize, ostream);

	if(analyzedStepCount)
		*analyzedStepCount = sampleSize;

	std::ofstream file(outputPath);
	file.precision(std::numeric_limits<double>::digits10 + 2);

	file << "#rundirs = [";
	bool isFirst = true;
	for(const auto& rundir : rundirs)
	{
		if(!isFirst)
			file << ", ";
		file << "'" << rundir << "'";
		isFirst = false;
	}
	file << "]\n";

	file << "#k_nx = " << k_n.getX() << "\n";
	file << "#k_ny = " << k_n.getY() << "\n";
	file << "#k_nz = " << k_n.getZ() << "\n";

	file << "#L_x = " << L_x << "\n";
	file << "#L_y = " << L_y << "\n";
	file << "#L_z = " << L_z << "\n";

	file << "#kT = " << kT << "\n";
	file << "#mpcTimestep = " << timestep << "\n";

	file << "#sampleSizes = " << sampleSize << "\n";

	file << "#\n";

	file << "#correlation-time" << "\t";
	file << "normalized-TVACFFT" << "\t";
	file << "TVACFFT" << "\t";
	file << "sample-size" << "\t";
	file << "TVACFFT-sample-standard-deviation" << "\t";
	file << "TVACFFT-DDDA-optimal-block-ID" << "\t";
	file << "TVACFFT-DDDA-optimal-standard-error-of-the-mean" << "\t";
	file << "TVACFFT-DDDA-optimal-standard-error-of-the-mean-is-reliable";
	file << "\n";

	double normalization = 0;
	for(const auto& point : tvacfft)
	{
		if(normalization == 0)
			normalization = point.second.getSampleMean();

		file << point.first << "\t";
		file << point.second.getSampleMean() / normalization << "\t";
		file << point.second.getSampleMean() << "\t";
		file << point.second.getSampleSize() << "\t";
		file << point.second.getSampleStandardDeviation() << "\t";
		file << point.second.getOptimalBlockIDForStandardErrorOfTheMean()
		     << "\t";
		file << point.second.getOptimalStandardErrorOfTheMean() << "\t";
		file << point.second.optimalStandardErrorOfTheMeanEstimateIsReliable();
		file << "\n";
	}
}

void writePlotfiles(
	const double minimumTime,
	const double maxCorrelationTime, const std::vector<std::string>& rundirs,
	const std::string& pathBasename,
	const std::string& metadataFilename,
	const std::string& metadataTableFilename,
	std::ostream* const ostream)
{
	const unsigned int simBoxX =
		MPCDAnalysis::Utilities::getConsistentConfigValue<unsigned int>(rundirs, "mpc.simulationBoxSize.x");
	const unsigned int simBoxY =
		MPCDAnalysis::Utilities::getConsistentConfigValue<unsigned int>(rundirs, "mpc.simulationBoxSize.y");
	const unsigned int simBoxZ =
		MPCDAnalysis::Utilities::getConsistentConfigValue<unsigned int>(rundirs, "mpc.simulationBoxSize.z");

	const double kT =
		MPCDAnalysis::Utilities::getConsistentConfigValue<double>(rundirs, "bulkThermostat.targetkT");

	const double mpcTimestep =
		MPCDAnalysis::Utilities::getConsistentConfigValue<double>(rundirs, "mpc.timestep");


	const unsigned int maxStride = ceil(maxCorrelationTime / mpcTimestep);


	std::size_t totalAnalyzedStepCount = 0;

	const auto waveVectorsAndPaths = getWaveVectorsAndPaths(rundirs);

	for(const auto& it : waveVectorsAndPaths)
	{
		const RealVector k_n = it.first;

		const double k_x = k_n.getX() * 2 * boost::math::constants::pi<double>() / simBoxX;
		const double k_y = k_n.getY() * 2 * boost::math::constants::pi<double>() / simBoxY;
		const double k_z = k_n.getZ() * 2 * boost::math::constants::pi<double>() / simBoxZ;

		const RealVector k(k_x, k_y, k_z);


		std::ostringstream ss;
		ss << pathBasename;
		ss << "--k_nx=" << k_n.getX();
		ss << "--k_ny=" << k_n.getY();
		ss << "--k_nz=" << k_n.getZ();
		ss << ".data";

		if(ostream)
			*ostream << "Processing k-vector (" << k_n << ")\n";

		std::size_t analyzedStepCount = 0;
		writePlotfile(
			it.second, ss.str(), k_n, k.getComplexVector(), kT,
			minimumTime, mpcTimestep,
			maxStride, rundirs, simBoxX, simBoxY, simBoxZ, ostream,
			&analyzedStepCount);
		totalAnalyzedStepCount += analyzedStepCount;

		if(ostream)
			*ostream << "\n";
	}

	if(!metadataFilename.empty())
	{
		std::ofstream file(metadataFilename);
		file << "{";

		file << "\"minimumTime\": " << minimumTime << ",\n";
		file << "\"totalAnalyzedStepCount\": "
		     << totalAnalyzedStepCount << ",\n";

		file << "\"runDirectories\": [\n";
		bool first = true;
		for(const auto& rundir : rundirs)
		{
			if(!first)
				file << ",\n";
			first = false;

			file << "\"" << rundir << "\"";
		}
		file << "]\n";

		file << "}";
	}

	if(!metadataTableFilename.empty())
	{
		std::ofstream file(metadataTableFilename);

		file << "minimumTime" << "\t";
		file << "totalAnalyzedStepCount" << "\n";


		file << minimumTime << "\t";
		file << totalAnalyzedStepCount << "\n";
	}
}
