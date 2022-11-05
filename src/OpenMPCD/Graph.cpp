#include <OpenMPCD/Graph.hpp>

#include <OpenMPCD/FilesystemUtilities.hpp>

#include <fstream>
#include <limits>

void OpenMPCD::Graph::save(const std::string& path, const bool prependGnuplotCommands) const
{
	FilesystemUtilities::ensureParentDirectory(path);

	std::ofstream file(path.c_str(), std::ios::trunc);
	file.precision(std::numeric_limits<FP>::digits10 + 2);

	if(prependGnuplotCommands)
		file<<"set terminal wxt persist\nplot '-' with linespoints notitle\n";

	for(Container::size_type i=0; i<points.size(); ++i)
		file<<points[i].first<<"\t"<<points[i].second<<"\n";
}
