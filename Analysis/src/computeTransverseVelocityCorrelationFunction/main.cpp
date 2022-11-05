#include <computeTransverseVelocityCorrelationFunction/functions.hpp>

#include <boost/program_options.hpp>

#include <iostream>

static int mymain(const int argc, const char* const argv[])
{
	namespace bpo = boost::program_options;

	const std::string programDescription(
		"Calculates the transverse velocity correlation function in the k-t-domain."
		);


	bpo::options_description visibleCommandLineOptions;
	bpo::options_description hiddenCommandLineOptions;
	bpo::options_description commandLineOptions("Allowed options");

	visibleCommandLineOptions.add_options()
		("help", "Prints help for this program.")
		("saveFilename", bpo::value<std::string>()->required(),
			"Filename, relative to the default directory, to save the processed data to")
		("metadataFilename", bpo::value<std::string>()->default_value(""),
			"If given and non-empty, saves analysis metadata to that path.")
		("metadataTableFilename", bpo::value<std::string>()->default_value(""),
			"If given and non-empty, saves a subset of analysis metadata to "
			"that path in tabular format.")
		("minimumTime",  bpo::value<double>()->default_value(0.0),
			"Data that have been recorded at times less than this argument are "
			"ignored. This can be used to account for some \"warm-up\" time.")
		("maxCorrelationTime", bpo::value<double>()->default_value(250),
			"Maximum correlation time to compute results for")
		("quiet", "Suppress output not indicative of problems.")
		;

	hiddenCommandLineOptions.add_options()
		("rundirs", bpo::value<std::vector<std::string> >()->required(),
			"Run directories to process.")
		;

	commandLineOptions.add(visibleCommandLineOptions);
	commandLineOptions.add(hiddenCommandLineOptions);

	bpo::positional_options_description positionals;
	positionals.add("rundirs", -1);

	bpo::variables_map varmap;
	bpo::store(
		bpo::command_line_parser(argc, argv)
			.options(commandLineOptions)
			.positional(positionals)
			.run(),
		varmap);

	if(varmap.count("help"))
	{
		std::cout << programDescription << "\n";
		std::cout << "\n";
		std::cout << "Usage: " << argv[0] << " rundir [ rundirs ... ]\n";
		std::cout << visibleCommandLineOptions << "\n";
		return 0;
	}

	bpo::notify(varmap);

	const std::vector<std::string> rundirs = varmap["rundirs"].as<std::vector<std::string> >();
	const std::string saveFilename = varmap["saveFilename"].as<std::string>();
	const std::string metadataFilename =
		varmap["metadataFilename"].as<std::string>();
	const std::string metadataTableFilename =
		varmap["metadataTableFilename"].as<std::string>();
	const double minimumTime = varmap["minimumTime"].as<double>();
	const double maxCorrelationTime = varmap["maxCorrelationTime"].as<double>();


	std::ostream* ostream = nullptr;
	if(varmap.count("quiet") == 0)
		ostream = &std::cout;

	writePlotfiles(
		minimumTime,
		maxCorrelationTime, rundirs, saveFilename,
		metadataFilename, metadataTableFilename,
		ostream);

	return 0;
}

int main(int argc, char* argv[])
{
	try
	{
		return mymain(argc, argv);
	}
	catch(const OpenMPCD::Exception& e)
	{
		std::cerr << "Caught exception:\n";
		std::cerr << e.getMessage() << "\n";
		return 1;
	}
	catch(const std::exception& e)
	{
		std::cerr << "Caught exception:\n";
		std::cerr << e.what() << "\n";
		return 1;
	}
}
