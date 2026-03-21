# NumClass — a mathematical X-ray scanner for integers

NumClass is a Python command-line tool that classifies integers into **200+ number-theory properties**.

Give it an integer and it can reveal properties spanning primes, divisors, figurate numbers, pseudoprimes, digit-based curiosities, named sequences, dynamical sequences, Diophantine representations, and more. It is designed for both everyday exploration and large-integer experiments, with support for configurable profiles, user extensions, and overrideable data files.

## Why NumClass?

Many integer tools answer one narrow question:

- Is this number prime?
- What is its factorization?
- Is it triangular or perfect?

NumClass takes a broader approach: it asks **what kind of number is this?**

That makes it useful for:

- recreational number theory
- OEIS-inspired exploration
- education and demos
- testing mathematical curiosities
- experimenting with custom classifiers
- exploring large integers in a structured way

## What it does

NumClass can identify and explain properties across many families, including:

- arithmetic and divisor-based numbers
- combinatorial and geometric sequences
- conjecture- and equation-based classes
- digit-based numbers
- Diophantine representations
- dynamical sequences
- figurate and polygonal numbers
- prime and prime-related numbers
- pseudoprimes and cryptographic numbers
- named sequences and mathematical curiosities
- “fun numbers” from computing, science fiction, and pop culture

It also includes:

- **200+ classifiers**
- **detailed explanations** for matches
- **intersection logic** for combined labels
- **profiles** to enable or disable groups of classifiers
- **workspace overrides** for data, profiles, and classifiers
- **custom classifier discovery**
- **OEIS-backed data files** where useful for heavy computations
- **aliquot sequence support**
- **large-integer friendly design**

## Installation

### Recommended: install as a CLI with `pipx`

```bash
pipx install "git+https://github.com/c788630/Numclass@main"
```

If you do not already have `pipx`:

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

On Windows, you can also use:

```bash
py -m pip install --user pipx
py -m pipx ensurepath
```

After `ensurepath`, restart your terminal if needed.

### Alternative: install with `pip`

```bash
pip install "git+https://github.com/c788630/Numclass@main"
```

## Quick start

Initialize your workspace:

```bash
numclass init
```

See where NumClass stores its workspace paths:

```bash
numclass where
```

Try it on a classic example:

```bash
numclass 1729
```

Run interactive mode:

```bash
numclass
```

## Typical usage

```bash
numclass [number]
```

If no number is provided, NumClass starts in interactive mode.

Useful options:

```bash
numclass --help
numclass --debug 1729
numclass --no-details 1729
numclass --quiet 1729
```

## Example
[Example output](https://github.com/c788630/Numclass/blob/main/images/output42.jpg)

## Workspace and customization

NumClass uses a user workspace so that you can customize behavior **without modifying the installed package**.

The workspace can contain:

- profiles
- data overrides
- user-defined classifiers
- intersections and other editable configuration

This design is intended to make NumClass a configurable classification framework, not just a fixed one-purpose script.

## Documentation

This README is the quick-start guide.

For the full user manual, examples, classifier details, and additional background, see the documentation in [`docs/`](docs/).

## Project structure

At a high level:

- `src/numclass/` - application code
- `tests/` - tests
- `docs/` - manual and extended documentation
- `images/` - screenshots and supporting images

## License

NumClass is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0).
