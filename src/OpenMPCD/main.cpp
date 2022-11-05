#include <OpenMPCD/AnalyticQuantities.hpp>
#include <OpenMPCD/AnalyticQuantitiesGaussianDumbbell.hpp>
#include <OpenMPCD/CUDA/Instrumentation.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/getGitCommitIdentifier.hpp>

#include <boost/chrono.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <ctime>
#include <fstream>
#include <signal.h>
#include <sstream>
#include <sys/stat.h>

using namespace OpenMPCD;

static const char* const defaultConfigPath="./config.txt";


static const std::string getCurrentDateTimeString()
{
	char datetimeString[512];
	const time_t datetime=std::time(0);
	struct tm* datetimeTM=localtime(&datetime);
	if(datetimeTM==NULL)
		OPENMPCD_THROW(Exception, "Failed to get current datetime.");

	if(strftime(datetimeString, sizeof(datetimeString), "%F_%T", datetimeTM)==0)
		OPENMPCD_THROW(Exception, "Failed to format current datetime.");

	return datetimeString;
}

static void createRundir(
	std::string* const rundir_ptr, std::string* const configPath_ptr)
{
	std::string& rundir=*rundir_ptr;

	if(rundir.empty())
	{
		rundir="runs/";
		rundir+=getCurrentDateTimeString();
	}

	if(boost::filesystem::exists(rundir))
	{
		typedef boost::filesystem::directory_iterator DI;
		for(DI it(rundir); it != DI(); ++it)
		{
			if(it->path().filename() == "config.txt")
			{
				*configPath_ptr = it->path().string();
			}
			else if(it->path().filename() == "input")
			{
			}
			else
			{
				OPENMPCD_THROW(Exception, "Specified rundir is non-empty.");
			}
		}
	}
	else
	{
		if(!boost::filesystem::create_directory(rundir))
			OPENMPCD_THROW(Exception, "Failed to create run directory.");
	}
}



static bool terminateProgram = false;
static bool earlySave = false;

#ifdef OPENMPCD_PLATFORM_POSIX
	static void signalHandler(int signum)
	{
		switch(signum)
		{
			case SIGHUP:
				break;

			case SIGINT:
			case SIGTERM:
				if(terminateProgram)
				{
					std::cout << "Multiple termination signals caught. "
							      "Aborting execution.\n";
					abort();
				}
				else
				{
					terminateProgram = true;
					std::cout << "Termination signal caught. "
							     "The simulation will gracefully end after the "
							      "current sweep.\n";
				}
				break;

			case SIGUSR1:
				earlySave = true;
				std::cout<<"Triggered early save..."<<std::flush;
				break;
		}
	}
#endif

static void installSignalHandler()
{
	#ifdef OPENMPCD_PLATFORM_POSIX
		struct sigaction signalAction;

		signalAction.sa_handler = &signalHandler;
		signalAction.sa_flags   = 0;
		if(sigemptyset(&signalAction.sa_mask)!=0)
		{
			OPENMPCD_THROW(
				Exception,
				"Failed to install signal handlers (sigemptyset). Aborting.\n");
		}

		int ret = 0;
		ret+=sigaction(SIGHUP, &signalAction, NULL);
		ret+=sigaction(SIGINT, &signalAction, NULL);
		ret+=sigaction(SIGTERM, &signalAction, NULL);
		ret+=sigaction(SIGUSR1, &signalAction, NULL);
		if(ret!=0)
		{
			OPENMPCD_THROW(
				Exception,
				"Failed to install signal handlers (sigaction). Aborting.\n");
		}
	#endif
}

static unsigned int generateSeed()
{
	const unsigned int seed_1 = std::time(0);
	const unsigned int seed_2 =
		boost::chrono::duration_cast<boost::chrono::nanoseconds>(
			boost::chrono::high_resolution_clock::now().time_since_epoch()
		).count();

	return seed_1 + seed_2;
}

static void run(const unsigned int maxRuntime, std::string rundir)
{
	installSignalHandler();


	std::string configPath = defaultConfigPath;
	createRundir(&rundir, &configPath);


	const unsigned int rngSeed = generateSeed();

	CUDA::Simulation sim(configPath, rngSeed, rundir);
	CUDA::Instrumentation instrumentation(&sim, rngSeed, getGitCommitIdentifier());
	instrumentation.setAutosave(rundir);

	const Configuration& config = sim.getConfiguration();

	const unsigned int sweepCount = config.read<unsigned int>("mpc.sweeps");

	const time_t startTime = std::time(0);

	sim.warmup();

	for(unsigned int i=0; i<sweepCount; ++i)
	{
		if(maxRuntime != 0)
		{
			if(std::time(0) - startTime > maxRuntime)
			{
				std::cout<<"Runtime limit has been reached. Terminating.\n";
				break;
			}
		}

		if(terminateProgram)
		{
			std::cout<<"Terminating prematurely.\n";
			break;
		}

		if(earlySave)
		{
			earlySave = false;

			try
			{
				const std::string earlyRundir = rundir + "/EarlySave_" + getCurrentDateTimeString();

				if(!boost::filesystem::create_directory(earlyRundir))
					OPENMPCD_THROW(IOException, "Failed to create early save directory: "+earlyRundir);

				instrumentation.save(earlyRundir);
				std::cout<<" done\n"<<std::flush;
			}
			catch(const Exception& e)
			{
				std::cerr<<"\n\n\t*** non-fatal exception ***\n";
				std::cerr<<"Encountered during early save:\n";
				std::cerr<<e.getMessage();
				std::cerr<<"\n\n";
			}
		}

		sim.sweep();
		instrumentation.measure();

		if(i%100==0)
			std::cout<<"Done with sweep "<<i<<"\n";
	}
}

int main(int argc, char** argv)
{
	try
	{
		namespace BPO = boost::program_options;
		namespace BPOS = BPO::command_line_style;

		unsigned int maxRuntime;
		std::string rundir;

		BPO::options_description cmdLineDesc("Supported command line options");
		cmdLineDesc.add_options()
			("help", "Print help message.")
			("maxRuntime", BPO::value<unsigned int>(&maxRuntime)->default_value(0),
				"The maximum (wall-time) runtime of the program, in seconds. 0 for no limit.")
			("rundir", BPO::value<std::string>(&rundir),
				"Specify the directory to save the run data in. If the "
				"directory does not exist, it will be created. If it does "
				"exist, it may at most contain the file `config.txt`, which "
				"then will be used as the input configuration file, and "
				"the directory may also contain an entry named `input`.");

		const BPO::positional_options_description positionalOptions;

		BPO::variables_map cmdLineVariables;
		BPO::store(
			BPO::command_line_parser(argc, argv).
				options(cmdLineDesc).
				positional(positionalOptions).
				style(
					BPOS::allow_short |
					BPOS::allow_dash_for_short |
					BPOS::short_allow_next |
					BPOS::allow_long |
					BPOS::long_allow_adjacent
					).
				run(),
			cmdLineVariables);
		BPO::notify(cmdLineVariables);

		if(cmdLineVariables.count("help"))
		{
			std::cout << cmdLineDesc << "\n";
			return 1;
		}


		if(maxRuntime != 0)
		{
			std::cout<<"Running for at most "<<maxRuntime<<" seconds.\n";
		}
		else
		{
			std::cout<<"No runtime bound specified.\n";
		}

		run(maxRuntime, rundir);
	}
	catch(const Exception& e)
	{
		std::cerr<<"\n\n\t*** EXCEPTION ***\n";
		std::cerr<<e.getMessage();
		std::cerr<<"\n\n";
		return 1;
	}
	catch(const boost::program_options::error& e)
	{
		std::cerr<<"\n\n\t*** EXCEPTION (boost::program_options::error) ***\n";
		std::cerr<<e.what();
		std::cerr<<"\n\n";
		std::cerr<<"Try invoking this program with the `--help` option.";
		std::cerr<<"\n\n";
	}
	catch(const std::exception& e)
	{
		std::cerr<<"\n\n\t*** EXCEPTION (std::exception) ***\n";
		std::cerr<<e.what();
		std::cerr<<"\n\n";
		return 2;
	}
	catch(...)
	{
		std::cerr<<"\n\n\t*** UNKNOWN EXCEPTION ***\n";
		return 3;
	}

	return 0;
}
