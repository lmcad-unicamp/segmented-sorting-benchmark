#include "array_generator_funcs.h"
#include "arglib.hpp"

#include <sys/resource.h> // getrusage() 

#include<fstream>

clarg::argString  ds_filename("-ds", "Input dataset filename.", "");
clarg::argDouble  pldist("-pldist", "Power-Law distribution alpha.", 0.0);
clarg::argBool    help("-h",  "display the usage message.");
clarg::argInt     maxbits("-maxbits", "Array values are randomly generated in the rage [0:2^maxbits[. Maxbits default = 12.", 12);
clarg::argInt     rpint("-rpint", "Report interval in seconds.", 30);

void show_usage(int argc, char** argv, ostream& os)
{
    os << std::endl << "Usage: " << argv[0] 
        << " -ds DATASET_FILE [other arguments] [-h]\n\n";

    os << "This program generates, for each dataset entry on DATASET_FILE,\n"
       << "an array with ARRAYSZ random numbers divided into NSEGS segments, sort it several\n"
       << "times using multiple strategies and report their execution time.\n";

    os << "The input dataset file must contain dataset description tuples, one per line.\n"
       << "Each dataset description tuple consists of a string with three values separeted\n" 
       << "by the `:' character (e.g. 10000:10:134). The first value especifies the array size,\n"
       << "the second one specifies the number of segments, and the third one specifies the\n"
       << "random number generator seed\n\n";

	os << "The program contains the following arguments:\n\n";

    clarg::arguments_descriptions(os, "  ", "\n");

    os << "\nIf -pldist ALPHA argument is provided, the segments sizes are generated using a power-law\n"
       << "distribution with alpha = ALPHA; \n"
       << "Otherwise, the segments size are made equal.\n\n";

    os << "Segmented sorting approaches may be individually enabled using -run* flags (e.g., -runNThrust flag).\n"
       << "the -runAll flag enables all of them.\n\n";

    os << "The -rpint flag specify the interval (in seconds) between progress report messages.\n\n";


    os << "Options -dumpSegmentSizes and -dryRun may be combined to generate the segment sizes.\n"
       << "generated for each dataset description tuple.\n\n";
}

int evaluate_strategies(int nsegments, int* segment_indices, int array_sz, int* array_values, int seed);

int initialize_array(int nsegments, int* segment_indices, int array_sz, int* array_values, int seed) 
{ 
    if (pldist.was_set()) {
        PowerLawRandomNumberGenerator rnd_generator(seed, pldist.get_value());
        if (segments_gen(rnd_generator, array_sz, nsegments, segment_indices)) {
            std::cerr << "ERROR: segments_gen(...) returned error" << std::endl;
            return 1;
        }
    }
    else {
        ConstantNumberGenerator num_generator(1);
        if (segments_gen(num_generator, array_sz, nsegments, segment_indices)) {
            std::cerr << "ERROR: segments_gen(...) returned error" << std::endl;
            return 2;
        }
    }

    UniformRandomNumberGenerator num_generator(seed);
    if (int_array_gen(num_generator, array_sz, maxbits.get_value(), array_values)) {
        std::cerr << "ERROR: int_array_gen(...) returned error" << std::endl;
        return 3;
    }

    return 0;
}

/* A test case is defined by an array size, the number of 
   segments and the random number generator seed. */
struct TestCase_t {
    TestCase_t(int sz, int ns, int sd) : arraysz(sz), nsegs(ns), seed(sd) {}
    int arraysz;
    int nsegs;
    int seed;
};

typedef vector< TestCase_t > TestCases_t;

int read_dataset(std::string filename, TestCases_t& testcases)
{
    std::fstream file(filename,std::ios_base::in);
    if (!file.is_open()) {
        std::cerr << "Could not open " << filename << std::endl;
        return -5;
    }

    int arraysz, nsegs, seed;
    char c1, c2;
    while (file >> arraysz >> c1 >> nsegs >> c2 >> seed) {
        testcases.push_back( TestCase_t(arraysz, nsegs, seed));
    }

    file.close();

    return 0; // OK
}

#include <sys/time.h>

/* Get time in seconds. */
double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + ( (double) tv.tv_usec / 1000000);
}

clarg::argBool dumpSegSizes ("-dumpSegmentSizes", "Dump the segment sizes.");

clarg::argBool dryRun ("-dryRun", "Read dataset file and build arrays, but does not run segment sorting kernels.");

int main(int argc, char** argv) 
{
    // Parse the arguments
    if (clarg::parse_arguments(argc, argv)) {
      cerr << "Error when parsing the arguments!" << endl;
      return 1;
    }
  
   if (help.get_value() == true) {
        show_usage(argc, argv, cerr);
        return 0;
   }

#define require_argument(arg) {                                    \
  if (!arg.was_set()) {                                            \
    cerr << arg.get_name() << " argument must be provided!";       \
    show_usage(argc, argv, cerr);                                  \
    return 1;                                                      \
  }                                                                \
}
    /* Check required arguments. */
    require_argument(ds_filename)

    TestCases_t testcases;
    if (read_dataset(ds_filename.get_value(), testcases)) {
        cerr << "ERROR: read_dataset() returned error." << std::endl;
        return -4;
    }

    double lastTime = getCurrentTime();
    double startTime = lastTime;
    int reportInterval = ((double) rpint.get_value() / 100.0) * testcases.size();

    /* For each element on the dataset descriptor, do: */ 
    for (int i=0; i < testcases.size(); i++) 
    {
        int array_sz  = testcases[i].arraysz;
        int nsegments = testcases[i].nsegs;
        int seed      = testcases[i].seed;

        if (nsegments > array_sz) {
            cerr << "ERROR: test case configuration error - number of segments "
                 << "must be less than or equal to the array size!\n";
            return 1;
        }

        /* Allocate and initilize array. */
        int* segment_indices = new int[nsegments+1];
        int* array           = new int[array_sz];

        if (initialize_array(nsegments, segment_indices, array_sz, array, seed)) { 
            cerr << "ERROR: generate_array() returned error." << std::endl;
            return -1;
        }

        if (dumpSegSizes.was_set()) {
            std::cout << array_sz << ":" << nsegments << ":" << seed << ":";
            for (int i=0; i<nsegments; i++) {
                if (i != 0) cout << ",";
                std::cout << segment_indices[i+1] - segment_indices[i];
            }
            std::cout << std::endl;
        }

        if (! dryRun.was_set()) {
            if (evaluate_strategies(nsegments, segment_indices, array_sz, array, seed)) { 
                cerr << "ERROR: evaluate_strategies() returned error." << std::endl;
                return -2;
            }
        }

        /* Free arrays. */
        delete[] segment_indices;
        delete[] array;

        /* Print progress report - [elapsed time]: i/total dataset configs concluded (%) - Max Resident Size so far*/
        double time = getCurrentTime();
        if ( (time - lastTime) > rpint.get_value()) {
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
                cerr << "[" << time - startTime << "]: " 
                    << i << "/" << testcases.size() << " concluded (" 
                    << 100.0 * (double) i / (double) testcases.size() << "%)"
                    << " - (Max RSS: " << usage.ru_maxrss / (1024) << "MB) \n";
            }
            else{
                cerr << "[" << time - startTime << "]: " 
                    << i << "/" << testcases.size() << " concluded (" 
                    << 100.0 * (double) i / (double) testcases.size() << "%)"
                    << " - (Max RSS: ? MB)\n";
            }
            lastTime = time;
        }

    }

    return 0;
}

