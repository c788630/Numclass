# 🧮 Number Classification Engine

This is a highly extensible and modular Python script that classifies integer numbers according to 8 main categories and 145 number classicications. It determines whether a number belongs to categories such as:

### Core Categories:
- **Arithmetic and Divisor-based:** perfect, abundant, deficient, powerful, practical, sociable, etc.
- **Digit-Based:** palindrome, automorphic, happy, narcissistic, self, Kaprekar, etc.
- **Integer Sequences:** Fibonacci, Bell, Catalan, Motzkin, Keith, Ulam, Taxicab, etc.
- **Polygonal and Figurate:** triangular, pentagonal, pronic, cyclic, repunit, etc.
- **Primes & Prime-related:** twin, safe, sexy, Sophie Germain, palindromic, emirp, factorial, strong, super, etc.
- **Pseudoprimes and Cryptographic numbers:** Carmichael, Fermat and Euler-Jacobi pseudoprimes, etc.
- **Mathematical Curiosities:** Cake, Munchausen, Kaprekar constants, strobogrammatic, vampire, Eban, etc.
- **Famous and fun numbers:** in pop culture, internet, computing, or sci-fi  

The classification is fast and accurate thanks to algorithmic optimizations and optional use of OEIS `bXXXXXX.txt` files.  
Inspired by [Numberphile](https://www.youtube.com/user/numberphile) and their love of surprising number facts!

## ▶️ Usage

```python numclass.py [-h] [--output OUTPUT] [--quiet] [--no-details] [--debug] [number]```

- `[number]` Number to classify (if no number is given, launches in interactive mode)

Options:
- `-h`, `--help` Show this help message and exit
- `--output OUTPUT` Output file or directory (see user/settings.py for more info)
- `--quiet` Suppress screen output (for quiet file output)
- `--no-details` Do not show explanation/details for results
- `--debug` Debug mode (including timings)

## 🚀 Features

- Over 140 mathematical classifications
- Efficient intersection logic (e.g., "Happy palindromic prime")
- Optional lookup acceleration via OEIS b-files
- Detailed explanations for each classification
- Auto-discovers custom classifiers: simply add your own!

Want to write your own property?  
1. Write a function: `def is_foobar_number(n): ...` in any `.py` file in `\classifiers`  
2. Add a `@classifier` decorator at the top of your function:
```
@classifier(  
    label="Some interesting property",
    description="description for your property",
    oeis="Axxxxxx", # OEIS number if applicable, else None
    category="your category" # or an existing one
)
```
3. That's it! Your classifier is auto-discovered at runtime.

## 📁 File Structure

```text
numclass.py                             # Main classification engine
utility.py                              # Utility functions for numclass.py
decorators.py                           # Decorator functions for numclass.py
output_manager.py                       # Output manager (screen and file)
test_numclass.py                        # Unit test set for all classifiers (pytest -v)
README.md                               # Documentation
LICENSE                                 # Licensing information
requirements.txt                        # requirements installer
classifiers/arithmetic_divisor.py       # Arithmetic and Divisor-based functions
classifiers/curiosities.py              # Mathematical curiosities
classifiers/digit_based.py              # Digit based test functions
classifiers/fun_number.py               # Fun numbers from pop culture, etc.
classifiers/polygonal_figurate.py       # Polygonal and Figurate Numbers
classifiers/prime.py                    # Prime and Prime-related functions
classifiers/pseudoprime_crypto.py       # Pseudoprime and cryptographic numbers
classifiers/sequences.py                # Integer sequences
data/b002093.txt                        # OEIS file for Highly abundant numbers
data/b004394.txt                        # OEIS file for Superabundant numbers
data/b005114.txt                        # OEIS file for Untouchable numbers
data/b104272.txt                        # OEIS file for Ramanujan primes
user/fun_numbers.py                     # Numberlist for fun_number.py
user/intersections.py                   # Intersection definitions
user/settings.py                        # Numclass settings
```

## 🛠 Dependencies

- Python 3.8+
- sympy 1.14 or later
- colorama 0.4.6 or later
- pytest 8.4.1 or later (only for testing test_numclass.py)
- OEIS `.txt` b-files for some optimized functions (e.g. superabundant, Keith numbers)  
You can download b-files from https://oeis.org/ by sequence number.  

Update your environment using: ```pip install -r requirements.txt```

## 🔍 Example Output

![numclass in action](images/output42.jpg)

## 🧠 Why?

Because every integer has a story.  
Whether you're a math enthusiast, educator, or Numberphile fan, this tool lets you explore the wild and wonderful world of numbers.

## ⚖️ License

Licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

> - Free to use, share, and adapt **for non-commercial purposes**.
> - **Attribution required** for use or modification.
> - Commercial use is **prohibited**.
> - Sequence data from the [OEIS Foundation](https://oeis.org/)

See: https://creativecommons.org/licenses/by-nc/4.0/

---
