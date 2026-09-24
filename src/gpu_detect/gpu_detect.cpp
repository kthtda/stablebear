#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#ifdef __linux__
#include <dirent.h>
#include <fstream>
#include <sstream>
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

std::vector<GpuInfo> detect_gpus()
{
  std::vector<GpuInfo> gpus;

  DIR* drm = opendir("/sys/class/drm");
  if (!drm)
    return gpus;

  const struct dirent* entry;
  while ((entry = readdir(drm)) != nullptr)
  {
    // Each GPU exposes several drm nodes (cardN, renderDN, connectors);
    // count only primary card nodes to avoid double-counting devices.
    const std::string node_name = entry->d_name;
    if (node_name.rfind("card", 0) != 0 || node_name.size() == 4 ||
        node_name.find_first_not_of("0123456789", 4) != std::string::npos)
    {
      continue;
    }

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
