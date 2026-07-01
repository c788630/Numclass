# Changelog

## [2.2.0] - 2026-07-01

### Added

#### Input transformation framework

* Added a new autodiscoverable text-to-integer transformation framework (`inputs/`).
* Input transformation modules are automatically discovered at startup.
* Added runtime reporting of loaded input transformations in debug mode.

#### New classifiers

* Added Prime-count landmark (A006880).
* Added Rooted tree number (A000081).
* Added Block-power invariant.
* Added Left factorial.
* Added Leyland number.
* Added Sierpiński candidate.
* Added Riesel candidate.
* Added Pernicious number.
* Added Graham's number suffix.
* Added Leyland prime intersection.

#### Number statistics

* Added prime counting function π(n) (up to 10^10).
* Added prime density statistics.

#### Data files

* Added `b006880.txt` (Prime-count landmark).
* Added `b000081.txt` (Rooted tree numbers).
* Added `graham_last_digits.txt` (Last 10,000 decimal digits of Graham's number).

#### Fun numbers

* Added God's number (20), the maximum number of moves required to solve any 3×3×3 Rubik’s Cube.
* Added 43,252,003,274,489,856,000, the number of reachable states of the Rubik’s Cube.

#### Expression evaluator

* Added subfactorial support.
* Added double factorial support.

### Improved

* Improved primorial classifier details.
* Improved Mersenne output for numbers larger than 100000 digits.
* Improved runtime and platform debug information.
* Improved default profile limits to keep calculations responsive.
* Added details text to the Feller number.
* Refactored the generic power-family framework.
* Help mode now accepts the Enter key for selection.
* Made `gmpy2` optional.

### iOS / a-Shell

* Added iOS runtime detection (`sys.platform == "ios"`).
* Added safe no-fork factorization backend for iOS.
* Fixed multiprocessing crashes in a-Shell during bounded factorization.
* Improved platform compatibility for large integer calculations.

### Fixed

* Corrected eBan number limits.
* Rewrote Automorphic number detection without string conversion.
* Fixed duplicate composite remainders causing incorrect exponents in factorizations.
* Fixed false semiprime classifications on incomplete factorizations.
* Fixed several large-integer edge cases related to Python integer string conversion limits.
* Fixed classifier discovery counting inconsistencies.
* Fixed duplicate classifier detection in data-driven classifiers.
* Fixed incorrect classifier totals in help output.
* Fixed input transformation reporting when no transformations are loaded.

### Notes

* Release 2.1.1 has been withdrawn due to an incorrect source ZIP asset.
