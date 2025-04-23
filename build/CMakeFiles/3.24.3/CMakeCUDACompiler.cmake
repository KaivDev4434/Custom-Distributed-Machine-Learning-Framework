set(CMAKE_CUDA_COMPILER "/apps/easybuild/software/CUDA/12.4.0/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/apps/easybuild/software/GCCcore/12.2.0/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.4.99")
set(CMAKE_CUDA_DEVICE_LINKER "/apps/easybuild/software/CUDA/12.4.0/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/apps/easybuild/software/CUDA/12.4.0/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "12.2")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/apps/easybuild/software/CUDA/12.4.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/apps/easybuild/software/CUDA/12.4.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "12.4.99")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/apps/easybuild/software/CUDA/12.4.0")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35-real;50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "80-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/apps/easybuild/software/CUDA/12.4.0/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/apps/easybuild/software/CUDA/12.4.0/targets/x86_64-linux/lib/stubs;/apps/easybuild/software/CUDA/12.4.0/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/apps/easybuild/software/libarchive/3.6.1-GCCcore-12.2.0/include;/apps/easybuild/software/cURL/7.86.0-GCCcore-12.2.0/include;/apps/easybuild/software/bzip2/1.0.8-GCCcore-12.2.0/include;/apps/easybuild/software/ncurses/6.3-GCCcore-12.2.0/include;/apps/easybuild/software/CUDA/12.0.0/nvvm/include;/apps/easybuild/software/CUDA/12.0.0/extras/CUPTI/include;/apps/easybuild/software/CUDA/12.0.0/include;/apps/easybuild/software/FFTW.MPI/3.3.10-gompi-2022b/include;/apps/easybuild/software/FFTW/3.3.10-GCC-12.2.0/include;/apps/easybuild/software/FlexiBLAS/3.2.1-GCC-12.2.0/include;/apps/easybuild/software/OpenBLAS/0.3.21-GCC-12.2.0/include;/apps/easybuild/software/OpenMPI/4.1.4-GCC-12.2.0/include;/apps/easybuild/software/UCC/1.1.0-GCCcore-12.2.0/include;/apps/easybuild/software/PMIx/4.2.2-GCCcore-12.2.0/include;/apps/easybuild/software/libfabric/1.16.1-GCCcore-12.2.0/include;/apps/easybuild/software/UCX/1.13.1-GCCcore-12.2.0/include;/apps/easybuild/software/libevent/2.1.12-GCCcore-12.2.0/include;/apps/easybuild/software/OpenSSL/1.1/include;/apps/easybuild/software/hwloc/2.8.0-GCCcore-12.2.0/include;/apps/easybuild/software/libpciaccess/0.17-GCCcore-12.2.0/include;/apps/easybuild/software/libxml2/2.10.3-GCCcore-12.2.0/include/libxml2;/apps/easybuild/software/libxml2/2.10.3-GCCcore-12.2.0/include;/apps/easybuild/software/XZ/5.2.7-GCCcore-12.2.0/include;/apps/easybuild/software/numactl/2.0.16-GCCcore-12.2.0/include;/apps/easybuild/software/binutils/2.39-GCCcore-12.2.0/include;/apps/easybuild/software/zlib/1.2.12-GCCcore-12.2.0/include;/apps/slurm/current/include;/mmfs1/apps/easybuild/software/GCCcore/12.2.0/include/c++/12.2.0;/mmfs1/apps/easybuild/software/GCCcore/12.2.0/include/c++/12.2.0/x86_64-pc-linux-gnu;/mmfs1/apps/easybuild/software/GCCcore/12.2.0/include/c++/12.2.0/backward;/mmfs1/apps/easybuild/software/GCCcore/12.2.0/lib/gcc/x86_64-pc-linux-gnu/12.2.0/include;/apps/easybuild/software/GCCcore/12.2.0/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/apps/easybuild/software/CUDA/12.4.0/targets/x86_64-linux/lib/stubs;/apps/easybuild/software/CUDA/12.4.0/targets/x86_64-linux/lib;/mmfs1/apps/easybuild/software/GCCcore/12.2.0/lib/gcc/x86_64-pc-linux-gnu/12.2.0;/mmfs1/apps/easybuild/software/GCCcore/12.2.0/lib/gcc;/apps/easybuild/software/libarchive/3.6.1-GCCcore-12.2.0/lib64;/apps/easybuild/software/cURL/7.86.0-GCCcore-12.2.0/lib64;/apps/easybuild/software/bzip2/1.0.8-GCCcore-12.2.0/lib64;/apps/easybuild/software/ncurses/6.3-GCCcore-12.2.0/lib64;/apps/easybuild/software/CUDA/12.0.0/stubs/lib64;/apps/easybuild/software/ScaLAPACK/2.2.0-gompi-2022b-fb/lib64;/apps/easybuild/software/FFTW.MPI/3.3.10-gompi-2022b/lib64;/apps/easybuild/software/FFTW/3.3.10-GCC-12.2.0/lib64;/apps/easybuild/software/FlexiBLAS/3.2.1-GCC-12.2.0/lib64;/apps/easybuild/software/OpenBLAS/0.3.21-GCC-12.2.0/lib64;/apps/easybuild/software/OpenMPI/4.1.4-GCC-12.2.0/lib64;/apps/easybuild/software/UCC/1.1.0-GCCcore-12.2.0/lib64;/apps/easybuild/software/PMIx/4.2.2-GCCcore-12.2.0/lib64;/apps/easybuild/software/libfabric/1.16.1-GCCcore-12.2.0/lib64;/apps/easybuild/software/UCX/1.13.1-GCCcore-12.2.0/lib64;/apps/easybuild/software/libevent/2.1.12-GCCcore-12.2.0/lib64;/apps/easybuild/software/OpenSSL/1.1/lib64;/apps/easybuild/software/hwloc/2.8.0-GCCcore-12.2.0/lib64;/apps/easybuild/software/libpciaccess/0.17-GCCcore-12.2.0/lib64;/apps/easybuild/software/libxml2/2.10.3-GCCcore-12.2.0/lib64;/apps/easybuild/software/XZ/5.2.7-GCCcore-12.2.0/lib64;/apps/easybuild/software/numactl/2.0.16-GCCcore-12.2.0/lib64;/apps/easybuild/software/binutils/2.39-GCCcore-12.2.0/lib64;/apps/easybuild/software/zlib/1.2.12-GCCcore-12.2.0/lib64;/mmfs1/apps/easybuild/software/GCCcore/12.2.0/lib64;/lib64;/usr/lib64;/apps/easybuild/software/libarchive/3.6.1-GCCcore-12.2.0/lib;/apps/easybuild/software/cURL/7.86.0-GCCcore-12.2.0/lib;/apps/easybuild/software/bzip2/1.0.8-GCCcore-12.2.0/lib;/apps/easybuild/software/ncurses/6.3-GCCcore-12.2.0/lib;/apps/easybuild/software/CUDA/12.0.0/lib;/apps/easybuild/software/ScaLAPACK/2.2.0-gompi-2022b-fb/lib;/apps/easybuild/software/FFTW.MPI/3.3.10-gompi-2022b/lib;/apps/easybuild/software/FFTW/3.3.10-GCC-12.2.0/lib;/apps/easybuild/software/FlexiBLAS/3.2.1-GCC-12.2.0/lib;/apps/easybuild/software/OpenBLAS/0.3.21-GCC-12.2.0/lib;/apps/easybuild/software/OpenMPI/4.1.4-GCC-12.2.0/lib;/apps/easybuild/software/UCC/1.1.0-GCCcore-12.2.0/lib;/apps/easybuild/software/PMIx/4.2.2-GCCcore-12.2.0/lib;/apps/easybuild/software/libfabric/1.16.1-GCCcore-12.2.0/lib;/apps/easybuild/software/UCX/1.13.1-GCCcore-12.2.0/lib;/apps/easybuild/software/libevent/2.1.12-GCCcore-12.2.0/lib;/apps/easybuild/software/OpenSSL/1.1/lib;/apps/easybuild/software/hwloc/2.8.0-GCCcore-12.2.0/lib;/apps/easybuild/software/libpciaccess/0.17-GCCcore-12.2.0/lib;/apps/easybuild/software/libxml2/2.10.3-GCCcore-12.2.0/lib;/apps/easybuild/software/XZ/5.2.7-GCCcore-12.2.0/lib;/apps/easybuild/software/numactl/2.0.16-GCCcore-12.2.0/lib;/apps/easybuild/software/binutils/2.39-GCCcore-12.2.0/lib;/apps/easybuild/software/zlib/1.2.12-GCCcore-12.2.0/lib;/mmfs1/apps/easybuild/software/GCCcore/12.2.0/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/apps/easybuild/software/binutils/2.40-GCCcore-13.2.0/bin/ld")
set(CMAKE_AR "/apps/easybuild/software/binutils/2.40-GCCcore-13.2.0/bin/ar")
set(CMAKE_MT "")
