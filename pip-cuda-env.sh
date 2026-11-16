# Put the CUDA toolkit installed with pip into the current Python environment
# (cuda-toolkit[nvcc,cudart]) first on PATH, so that nvcc and CMake use it
# instead of a system toolkit. Source it after activating the environment:
#
#   source pip-cuda-env.sh
#
# CUDAToolkit_ROOT stays unset: CMake finds the pip toolkit on its own and
# skips that search when the variable is set.

_sb_cuda_root=$(python -c "
import importlib.util
for pkg in ('nvidia.cu13', 'nvidia.cu12'):
    try:
        spec = importlib.util.find_spec(pkg)
    except ImportError:
        continue
    if spec and spec.submodule_search_locations:
        print(list(spec.submodule_search_locations)[0])
        break
" 2>/dev/null)

if [ -n "$_sb_cuda_root" ] && [ -x "$_sb_cuda_root/bin/nvcc" ]; then
    export PATH="$_sb_cuda_root/bin:$PATH"
    export CUDACXX="$_sb_cuda_root/bin/nvcc"
    echo "Using $("$CUDACXX" --version | tail -1) from $_sb_cuda_root"
else
    echo "No pip-installed nvcc in $(command -v python); pip install \"cuda-toolkit[nvcc,cudart]\"" >&2
fi
unset _sb_cuda_root
