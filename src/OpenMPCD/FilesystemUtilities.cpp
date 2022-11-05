#include <OpenMPCD/FilesystemUtilities.hpp>

#include <OpenMPCD/Exceptions.hpp>

#include <boost/filesystem.hpp>

using namespace OpenMPCD;

void FilesystemUtilities::ensureDirectory(const std::string& path)
{
	boost::system::error_code ec;
	boost::filesystem::create_directories(path, ec);

	if(ec)
		OPENMPCD_THROW(IOException, "Failed to create directory.");
}

void FilesystemUtilities::ensureParentDirectory(const std::string& path)
{
	boost::filesystem::path p(path);

	if(!p.has_parent_path())
		return;

	boost::system::error_code ec;
	boost::filesystem::create_directories(p.parent_path(), ec);

	if(ec)
		OPENMPCD_THROW(IOException, "Failed to create directory.");
}
