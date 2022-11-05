#include <OpenMPCD/CUDA/MPCSolute/Instrumentation/StarPolymers.hpp>

#include <OpenMPCD/FilesystemUtilities.hpp>

#include <boost/filesystem.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{
namespace Instrumentation
{

template<typename PositionCoordinate, typename VelocityCoordinate>
StarPolymers<PositionCoordinate, VelocityCoordinate>::StarPolymers(
	MPCSolute::StarPolymers<PositionCoordinate, VelocityCoordinate>* const
		starPolymers_,
	const Configuration::Setting& settings_)
	: starPolymers(starPolymers_), settings(settings_)
{
	resetSnapshotFile();
}

template<typename PositionCoordinate, typename VelocityCoordinate>
StarPolymers<PositionCoordinate, VelocityCoordinate>::~StarPolymers()
{
	if(!snapshotFilePath.empty())
		boost::filesystem::remove(snapshotFilePath);
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void StarPolymers<PositionCoordinate, VelocityCoordinate>::measureSpecific()
{
	starPolymers->writeStateToSnapshot(snapshotFile.get());
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void StarPolymers<PositionCoordinate, VelocityCoordinate>::saveSpecific(
	const std::string& rundir) const
{
	const std::string basedir = rundir + "/StarPolymers";
	FilesystemUtilities::ensureDirectory(basedir);

	if(settings.has("initialState"))
	{
		boost::filesystem::copy(
			settings.read<std::string>("initialState"),
			basedir + "/initialState.vtf");
	}

	snapshotFile.reset();
	boost::filesystem::copy(snapshotFilePath, basedir + "/snapshots.vtf");
	resetSnapshotFile();
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void
StarPolymers<PositionCoordinate, VelocityCoordinate>::resetSnapshotFile()
const
{
	if(!snapshotFilePath.empty())
		boost::filesystem::remove(snapshotFilePath);

	{
		const boost::filesystem::path tmp =
			boost::filesystem::temp_directory_path() /
			boost::filesystem::unique_path();
		snapshotFilePath = tmp.string();
	}
	snapshotFile.reset(new VTFSnapshotFile(snapshotFilePath));
	starPolymers->writeStructureToSnapshot(snapshotFile.get());
}

} //namespace Instrumentation
} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

template class
	OpenMPCD::CUDA::MPCSolute::Instrumentation::StarPolymers<double, double>;
