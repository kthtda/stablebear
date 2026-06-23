// Generic host-side data manager for variable-length objects.
// Pure C++ — no CUDA dependency.
//
// Manages N objects, each with a variable number of elements of type ElementT,
// flattened into a contiguous array with an offset index.

#ifndef SB_OFFSET_DATA_MANAGER_HPP
#define SB_OFFSET_DATA_MANAGER_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

namespace sb
{
  namespace internal
  {
    /// Host-side storage for N variable-length objects
    /// flattened into a contiguous element array with an offset index.
    template <typename ElementT>
    struct HostOffsetData
    {
      std::vector<size_t> offsets;
      std::vector<ElementT> elements;
    };
  }

  template <typename ElementT>
  class OffsetDataManager
  {
  public:
    OffsetDataManager() = default;

    /// Flatten a range of objects into the host-side offset + element arrays.
    ///
    /// @param begin, end   Forward iterators over the source objects.
    /// @param sizeFn       Callable: (const Obj&) -> size_t.
    ///                     Returns the number of elements for one object.
    /// @param elementFn    Callable: (const Obj&, size_t i) -> ElementT.
    ///                     Returns the i-th element for one object.
    template <typename FwdIt, typename SizeFn, typename ElementFn>
    void init(FwdIt begin, FwdIt end, SizeFn sizeFn, ElementFn elementFn)
    {
      m_nObjects = static_cast<size_t>(std::distance(begin, end));
      m_hostData.offsets.resize(m_nObjects + 1);

      size_t offset = 0;
      size_t i = 0;
      for (auto it = begin; it != end; ++it)
      {
        m_hostData.offsets[i++] = offset;
        offset += sizeFn(*it);
      }
      m_hostData.offsets[m_nObjects] = offset;

      m_hostData.elements.resize(offset);
      i = 0;
      for (auto it = begin; it != end; ++it)
      {
        size_t n = sizeFn(*it);
        auto coffs = m_hostData.offsets[i++];
        for (size_t j = 0; j < n; ++j)
        {
          m_hostData.elements[coffs + j] = elementFn(*it, j);
        }
      }
    }

    [[nodiscard]] size_t total_elements_for_range(size_t start, size_t count) const
    {
      return m_hostData.offsets[start + count] - m_hostData.offsets[start];
    }

    [[nodiscard]] size_t max_elements_in_range(size_t start, size_t count) const
    {
      size_t maxElems = 0;
      for (size_t i = start; i < start + count; ++i)
      {
        size_t n = m_hostData.offsets[i + 1] - m_hostData.offsets[i];
        maxElems = std::max(maxElems, n);
      }
      return maxElems;
    }

    [[nodiscard]] const internal::HostOffsetData<ElementT>& host_data() const { return m_hostData; }
    [[nodiscard]] size_t num_objects() const { return m_nObjects; }

  private:
    internal::HostOffsetData<ElementT> m_hostData;
    size_t m_nObjects = 0;
  };
}

#endif
