# Set these two variables based on your environment
export CUDA_HOME="$1"
export ARCH="$2"

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64  

if [ ! -d "${CUDA_HOME}" ]; then
    echo "ERROR: Invalid directory set for CUDA_HOME: \"${CUDA_HOME}\""
    return 1 2> /dev/null || exit 1 # Do not exit if sourcing this script (return)
fi

if [ ! -d "${CUDA_HOME}/lib64" ]; then
    echo "ERROR: CUDA_HOME directory contains no lib64 subdir: \"${CUDA_HOME}/lib64\""
    return 1 2> /dev/null || exit 1
fi

PATH=${CUDA_HOME}/bin:${PATH}

