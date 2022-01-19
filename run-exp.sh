
systemid=$1

DATASET=datasets/parallel-computing.ds

system_setup_script="systems/config_environment-${systemid}.sh"

if [ "${systemid}x" == "x" ]; then
    echo "ERROR: you must provide a system identifier."
    echo "Usage: $0 systemid"
    echo "There must exist a file configure_environment-systemid.sh on the systems directory"
    echo "List of files at systems directory:"
    ls -w1 systems/
    exit 1
fi
	
if [ ! -f "${system_setup_script}" ]; then
    echo "ERROR: ${system_setup_script} is not a valid file"
    exit 1
fi

basedir=`echo "$(pwd)"`
DT=`date "+%Y-%m-%d-%H-%M-%S"`
RESULTSDIR="${basedir}/results/${systemid}-${DT}"
mkdir -p "${RESULTSDIR}"
logfile="${RESULTSDIR}/run-exp.log"
echo "" > "${logfile}" # Clean log file

echo "Redirecting all outputs to log file: ${logfile}"

function report {
    echo "$@"
    echo "$@" >> "${logfile}"
}

function fail {
    report "FAIL: $@"
    D=`date`
    report "Failure time: $D"
    exit 1
}

D=`date`
report "Start time: $D"

echo "SYSTEM_SETUP_SCRIPT: \"${system_setup_script}\""
source "${system_setup_script}" 2>&1 >> "${logfile}" || \
   fail "Error when sourcing configuration script \"${system_setup_script}\""

# Build utils
report "1-Building utils..."
(cd utils && make || fail "Could not build utils") 2>&1 >> "${logfile}"

# Build sorting apps
report "2-Building sorting apps..."
(cd src && make segsort-benchmark.x || fail "Could not build segsort-benchmark.x")  2>&1 >> "${logfile}"

# Try to read the devices information
report "3-Collecting system info..."
report "lscpi"
lspci &> ${RESULTSDIR}/system-lspci.log
if [ -f utils/deviceQuery.x ]; then
    report "./utils/deviceQuery.x"
    ./utils/deviceQuery.x &> ${RESULTSDIR}/system-deviceQuery.log
fi
report "cat /proc/cpuinfo"
cat /proc/cpuinfo &> ${RESULTSDIR}/system-cpuinfo.log

report "cat /proc/meminfo"
cat /proc/meminfo &> ${RESULTSDIR}/system-meminfo.log

report "cat /etc/os-release"
cat /etc/os-release &> ${RESULTSDIR}/system-os-release.log

report "lsb_release -a"
lsb_release -a  &> ${RESULTSDIR}/system-lsb_release.log

report "hostnamectl"
hostnamectl &> ${RESULTSDIR}/system-hostnamectl.log

report "nvcc --version"
nvcc --version &> ${RESULTSDIR}/system-nvcc-version.log

# Executing benchmarks
report "4-Executing benchmark..."

ERROR_CODES=""
for alpha in -5.0 -0.9 0.0 0.9 5.0; do
    report "  - Evaluating strategies for alpha = ${alpha}"
    ./src/segsort-benchmark.x -ds ${DATASET} -pldist ${alpha} -runAll -chkResult 1> ${RESULTSDIR}/results.${alpha}.txt 2> ${RESULTSDIR}/results.${alpha}.log
    ERROR_CODES="${ERROR_CODES}, Alpha ${alpha} = $?"
done

report "  - Evaluating strategies for same sized segments (equal)"
./src/segsort-benchmark.x -ds ${DATASET} -runAll -chkResult 1> ${RESULTSDIR}/results.equal.txt 2> ${RESULTSDIR}/results.equal.log
ERROR_CODES="${ERROR_CODES}, equal = $?"

report "5-Execution finished with ERROR codes = ${ERROR_CODES}" 

D=`date`

report "End time: $D"
