#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#ifdef __linux__
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")
#endif

namespace py = pybind11;

namespace
{

static constexpr uint32_t NVIDIA_VENDOR_ID = 0x10de;

struct GpuInfo
{
  std::string name;
  uint32_t vendor_id;
  uint32_t device_id;
};

#ifdef __linux__

// Each physical GPU exposes several /sys/class/drm nodes (cardN, renderDN,
// and per-connector nodes that resolve to the same device). Count only the
// cardN nodes so one GPU is reported once.
bool is_drm_card_entry(const char* name)
{
  if (std::strncmp(name, "card", 4) != 0 || name[4] == '\0')
    return false;
  for (const char* p = name + 4; *p != '\0'; ++p)
  {
    if (!std::isdigit(static_cast<unsigned char>(*p)))
      return false;
  }
  return true;
}

std::vector<GpuInfo> detect_gpus_drm()
{
  std::vector<GpuInfo> gpus;

  DIR* drm = opendir("/sys/class/drm");
  if (!drm)
    return gpus;

  const struct dirent* entry;
  while ((entry = readdir(drm)) != nullptr)
  {
    if (!is_drm_card_entry(entry->d_name))
      continue;

    std::string base = "/sys/class/drm/";
    base += entry->d_name;
    std::string device_path = base + "/device/";

    // Read vendor ID
    std::string vendor_path = device_path + "vendor";
    std::ifstream vendor_file(vendor_path);
    if (!vendor_file.is_open())
      continue;

    uint32_t vendor_id = 0;
    vendor_file >> std::hex >> vendor_id;
    vendor_file.close();

    if (vendor_id != NVIDIA_VENDOR_ID)
      continue;

    GpuInfo gpu;
    gpu.vendor_id = vendor_id;
    gpu.device_id = 0;

    // Read device ID
    std::string devid_path = device_path + "device";
    std::ifstream devid_file(devid_path);
    if (devid_file.is_open())
    {
      devid_file >> std::hex >> gpu.device_id;
      devid_file.close();
    }

    // Try to get device name from uevent
    gpu.name = "NVIDIA GPU";
    std::string uevent_path = device_path + "uevent";
    std::ifstream uevent_file(uevent_path);
    if (uevent_file.is_open())
    {
      std::string line;
      while (std::getline(uevent_file, line))
      {
        if (line.rfind("PCI_SLOT_NAME=", 0) == 0)
        {
          // Use PCI slot as part of the identifier
          gpu.name = "NVIDIA GPU [" + line.substr(14) + "]";
          break;
        }
      }
      uevent_file.close();
    }

    gpus.push_back(std::move(gpu));
  }

  closedir(drm);
  return gpus;
}

// The driver's procfs has one directory per GPU. Covers slim containers
// with no lspci and no NVIDIA PCI drm entries.
std::vector<GpuInfo> detect_gpus_proc_driver()
{
  std::vector<GpuInfo> gpus;

  DIR* dir = opendir("/proc/driver/nvidia/gpus");
  if (!dir)
    return gpus;

  const struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr)
  {
    std::string name = entry->d_name;
    if (name == "." || name == "..")
      continue;

    GpuInfo gpu;
    gpu.vendor_id = NVIDIA_VENDOR_ID;
    gpu.device_id = 0;
    gpu.name = "NVIDIA GPU [" + name + "]";

    std::ifstream info_file("/proc/driver/nvidia/gpus/" + name + "/information");
    std::string line;
    while (std::getline(info_file, line))
    {
      if (line.rfind("Model:", 0) == 0)
      {
        auto pos = line.find_first_not_of(" \t", 6);
        if (pos != std::string::npos)
        {
          gpu.name = line.substr(pos);
        }
        break;
      }
    }

    gpus.push_back(std::move(gpu));
  }

  closedir(dir);
  return gpus;
}

// WSL2 exposes the GPU through /dev/dxg with no NVIDIA PCI drm nodes.
// /dev/dxg alone could be any vendor, so require the WSL NVIDIA driver
// libraries as well.
std::vector<GpuInfo> detect_gpus_wsl()
{
  std::vector<GpuInfo> gpus;

  if (access("/dev/dxg", F_OK) != 0)
    return gpus;

  const char* wsl_markers[] = {
    "/usr/lib/wsl/lib/libcuda.so.1",
    "/usr/lib/wsl/lib/libcuda.so",
    "/usr/lib/wsl/lib/nvidia-smi",
  };
  for (const char* marker : wsl_markers)
  {
    if (access(marker, F_OK) == 0)
    {
      GpuInfo gpu;
      gpu.vendor_id = NVIDIA_VENDOR_ID;
      gpu.device_id = 0;
      gpu.name = "NVIDIA GPU (WSL2)";
      gpus.push_back(std::move(gpu));
      break;
    }
  }

  return gpus;
}

std::vector<GpuInfo> detect_gpus()
{
  auto gpus = detect_gpus_drm();
  if (!gpus.empty())
    return gpus;

  gpus = detect_gpus_proc_driver();
  if (!gpus.empty())
    return gpus;

  return detect_gpus_wsl();
}

#elif defined(_WIN32)

std::vector<GpuInfo> detect_gpus()
{
  std::vector<GpuInfo> gpus;

  IDXGIFactory* factory = nullptr;
  HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory));
  if (FAILED(hr))
    return gpus;

  IDXGIAdapter* adapter = nullptr;
  for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
  {
    DXGI_ADAPTER_DESC desc;
    if (SUCCEEDED(adapter->GetDesc(&desc)))
    {
      if (desc.VendorId == NVIDIA_VENDOR_ID)
      {
        GpuInfo gpu;
        gpu.vendor_id = desc.VendorId;
        gpu.device_id = desc.DeviceId;

        // Convert wide string to narrow
        char name_buf[256];
        size_t converted = 0;
        wcstombs_s(&converted, name_buf, sizeof(name_buf), desc.Description, _TRUNCATE);
        gpu.name = name_buf;

        gpus.push_back(std::move(gpu));
      }
    }
    adapter->Release();
  }

  factory->Release();
  return gpus;
}

#else

std::vector<GpuInfo> detect_gpus()
{
  return {};
}

#endif

} // anonymous namespace


PYBIND11_MODULE(_gpu_detect, m)
{
  m.doc() = "Detect NVIDIA GPUs without CUDA dependencies";

  py::class_<GpuInfo>(m, "GpuInfo")
    .def_readonly("name", &GpuInfo::name)
    .def_readonly("vendor_id", &GpuInfo::vendor_id)
    .def_readonly("device_id", &GpuInfo::device_id)
    .def("__repr__", [](const GpuInfo& g) {
      return "GpuInfo(name='" + g.name + "', device_id=0x" +
        ([](uint32_t v) {
          char buf[16];
          snprintf(buf, sizeof(buf), "%04x", v);
          return std::string(buf);
        })(g.device_id) + ")";
    });

  m.def("detect_nvidia_gpus", &detect_gpus,
    "Detect NVIDIA GPUs using OS-level APIs (no CUDA required).\n"
    "Returns a list of GpuInfo objects.");

  m.def("has_nvidia_gpu", []() { return !detect_gpus().empty(); },
    "Returns True if at least one NVIDIA GPU is detected.");

  m.def("nvidia_gpu_count", []() { return detect_gpus().size(); },
    "Returns the number of NVIDIA GPUs detected.");
}
