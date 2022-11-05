#include <OpenMPCD/Graph4D.hpp>

#include <fstream>
#include <limits>

void OpenMPCD::Graph4D::save(const std::string& path) const
{
	std::ofstream file(path.c_str(), std::ios::trunc);
	file.precision(std::numeric_limits<FP>::digits10 + 2);

	for(Container::size_type i=0; i<points.size(); ++i)
		file<<points[i].get<0>()<<"\t"<<points[i].get<1>()<<"\t"<<points[i].get<2>()<<"\t"<<points[i].get<3>()<<"\n";
}
