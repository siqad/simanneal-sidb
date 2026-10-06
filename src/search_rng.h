#ifndef SIMANNEAL_SEARCH_RNG_H
#define SIMANNEAL_SEARCH_RNG_H
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
/* PCG XSH-RR algorithm and seeding adapted from pcg-c-basic/pcg_basic.c.
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modified for SimAnneal: C++ engine wrapper, explicit uniform grid,
 * and separate Lemire bounded sampler. No global engine or shared state.
 * Original: https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c
 */
namespace simanneal_rng {
struct Pcg32 {
  typedef std::uint32_t result_type;
  std::uint64_t state = 0, inc = 0;
  explicit Pcg32(std::uint64_t seed = 0) { reseed(seed, 54); }
  void reseed(std::uint64_t seed, std::uint64_t stream) {
    state = 0;
    inc = (stream << 1u) | 1u;
    (*this)();
    state += seed;
    (*this)();
  }
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return UINT32_MAX; }
  result_type operator()() {
    const std::uint64_t old = state;
    state = old * UINT64_C(6364136223846793005) + inc;
    const std::uint32_t shifted =
        static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
    const std::uint32_t rot = static_cast<std::uint32_t>(old >> 59u);
    return (shifted >> rot) | (shifted << ((-rot) & 31u));
  }
};
// Lemire 2019, arXiv1805.10941. Assumes full-range uniform 32-bit engine.
// bound==0 is deliberately disallowed; range 2^32 needs the raw engine.
template <class Engine>
inline std::uint32_t bounded(Engine &engine, std::uint32_t bound) {
  if (bound == 0)
    throw std::invalid_argument("bounded random range must be positive");
  std::uint64_t product = std::uint64_t(engine()) * bound;
  std::uint32_t low = static_cast<std::uint32_t>(product);
  if (low < bound) {
    const std::uint32_t threshold = (std::uint32_t(0) - bound) % bound;
    while (low < threshold) {
      product = std::uint64_t(engine()) * bound;
      low = static_cast<std::uint32_t>(product);
    }
  }
  return static_cast<std::uint32_t>(product >> 32u);
}
// Both supported engines have a finite uniform grid. Preserve its zero
// endpoint.
inline bool populationAcceptance(double draw, double x, double kT,
                                 bool shortcut) {
  const double bound = shortcut && std::isfinite(kT) && kT > 0 ? 40 * kT : 0;
  return bound > 0 && x > bound    ? draw == 0
         : bound > 0 && x < -bound ? true
                                   : draw <= 1. / (1 + std::exp(x / kT));
}
// The default specialization preserves the original division and shortcuts.
// Dispatch once per population update, with one shared sampling body.
template <bool Cached> struct PopulationProbability;
template <> struct PopulationProbability<false> {
  double temperature;
  bool shortcut;
  PopulationProbability(double kT, bool shortcuts)
      : temperature(kT), shortcut(shortcuts) {}
  bool accept(double draw, double x) const {
    return populationAcceptance(draw, x, temperature, shortcut);
  }
};
template <> struct PopulationProbability<true> {
  double temperature, bound, inverse;
  bool use_inverse;
  PopulationProbability(double kT, bool shortcut)
      : temperature(kT),
        bound(shortcut && std::isfinite(kT) && kT > 0 ? 40 * kT : 0),
        inverse(std::isnormal(kT) && kT > 0 ? 1. / kT : 0),
        use_inverse(std::isnormal(kT) && kT > 0 && std::isfinite(inverse)) {}
  bool accept(double draw, double x) const {
    return bound > 0 && x > bound ? draw == 0
         : bound > 0 && x < -bound ? true
         : draw <= 1. / (1 + std::exp(use_inverse ? x * inverse
                                                : x / temperature));
  }
};
inline bool hopAcceptance(double draw, double delta, double kT, bool shortcut) {
  const double bound = shortcut && std::isfinite(kT) && kT > 0 ? 40 * kT : 0;
  return bound > 0 && delta > bound ? draw == 0 : draw <= std::exp(-delta / kT);
}
struct UniformGrid32 {
  UniformGrid32(double = 0, double = 1) {}
  template <class Engine> double operator()(Engine &engine) const {
    return static_cast<double>(engine()) * (1.0 / 4294967296.0);
  }
};
} // namespace simanneal_rng
#endif
