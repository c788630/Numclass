# Changelog

## [1.1.0] - 2025-08-15
### Added
- Clear screen to different help sections
- New number statistics:
  - Additive persistence
  - Multiplicative persistence
  - 1/n (base 10) (repeat/terminate with period & preperiod)
  - Möbius μ(n)
  - Radical rad(n)
- Total solutions display for `is_sum_of_2_squares`
- More known solutions for `is_sum_of_3_cubes`
- New classifier (curiosities): Additive sequence

### Updated
- `is_sum_of_3_squares` and `is_sum_of_2_cubes` now show number of found solutions

### Fixed
- Line wrapping for Euler’s totient and Carmichael in Number statistics
- Added `__init__.py` to main folder

---

## [1.0.1] - 2025-08-09
### Added
- Euler’s totient calculation
- Carmichael calculation
- Absolute prime classifier
- Chen prime classifier
- Check on forbidden output filenames in settings and command line

### Updated
- Proth prime detection
- Gaussian prime detection

### Fixed
- Gaussian prime bug
- Carmichael number bug
- Aliquot typo
