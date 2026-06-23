#ifndef SB_PLATFORM_H
#define SB_PLATFORM_H

  #ifndef BUILD_WITH_CUDA

    #ifndef __host__
      #define __host__
    #endif

    #ifndef __device__
      #define __device__
    #endif

  #endif

  #ifdef _WIN32
    #define SB_EXPORT_API __declspec(dllexport)
  #else
    #define SB_EXPORT_API
  #endif

#endif
