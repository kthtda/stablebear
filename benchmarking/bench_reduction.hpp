#ifndef BENCH_REDUCTION_H
#define BENCH_REDUCTION_H

#include "benchmark.hpp"

#include <sbear/functional/pcf.hpp>
#include <vector>

class BenchmarkReduction : public Benchmark
{
public:
  void init(const boost::program_options::variables_map &) override;
  void run() override;
private:
  std::vector<sb::Pcf_f32> m_pcfs32;
  size_t m_chunksz = 2ul;
};

#endif
