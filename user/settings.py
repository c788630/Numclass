# settings.py

# ===============================================================
# Performance & Calculation Settings
# ===============================================================

# Allow slow (potentially unbounded) calculations?
# True: disables calculation timeouts and size limits for number properties.
# WARNING: This may be extremely slow or cause memory issues for very large numbers!
# False (default): restricts calculations to return results within seconds.
ALLOW_SLOW_CALCULATIONS = False

# Brute-force calculation limits (for "sum of X squares/cubes" etc.)
SUM_MIN = -100
SUM_MAX = 100

# Maximum number of steps in an Aliquot sequence calculation.
MAX_ALIQUOT_STEPS = 250

# Limit the number of results shown for all "sum_of" functions.
# None = unlimited; 1 = first solution only; n = up to n solutions.
MAX_SOLUTIONS_SUM_OF = {
    "2_SQUARES": None,                 
    "3_SQUARES": 20,
    "2_CUBES": None,
    "3_CUBES": 20,
}

# Maximum results for sum-of-palindromes decompositions.
# None = unlimited
MAX_SOLUTIONS_SUM_OF_PALINDROMES = {
    "2": None,   # Maximum results for sum of 2 palindromes.
    "3": 20,     # Maximum results for sum of 3 palindromes.
}

# Minimum number of digits for each palindrome in sum decompositions.
MIN_PALINDROME_DIGITS = 2
# Allow decompositions using zero for any sum-of-powers function
ALLOW_ZERO_IN_DECOMP = False
# The bounds for the search in sum of 3 qubes
MAX_ABS_FOR_SUM_OF_3_CUBES = 150

# ===============================================================
# Category and Classifier Settings
# ===============================================================

# Enable/disable entire classification categories.
# To add your own, create a new file in 'classifiers/' and set CATEGORY = "..."
# Make sure the name matches the key below.
CATEGORIES = {
    "Arithmetic and Divisor-based": True,
    "Digit-based": True,
    "Fun numbers": True,
    "Integer Sequences": True,
    "Mathematical Curiosities": True,
    "Polygonal and Figurate Numbers": True,
    "Primes & Prime-related Numbers": True,
    "Pseudoprimes and Cryptographic Numbers": True,
}

# Enable/disable individual classifiers (by label, all-caps with underscores).
# Useful to speed up classification with huge numbers.
# Example: To disable slow classifiers like "ULAM_NUMBER" or
# "UNTOUCHABLE_NUMBER", set it to False.
CLASSIFIERS = {
    "AMICABLE_NUMBER": True,
    "SOCIABLE_NUMBER": True,
    "SUM_OF_2_SQUARES": True,
    "SUM_OF_3_SQUARES": True,
    "SUM_OF_2_CUBES": True,
    "SUM_OF_3_CUBES": True,
    "SUM_OF_2_PALINDROMES": True,
    "SUM_OF_3_PALINDROMES": True,
    "ULAM_NUMBER": True,
    "UNTOUCHABLE_NUMBER": True,
    # Add/remove classifiers as you like!
}

# ===============================================================
# Output and Display Settings
# ===============================================================

# Show extra explanations for each classifier result?
SHOW_CLASSIFIER_DETAILS = True

# Show the full list of divisors for each number?
SHOW_DIVISORS = True   

# Limit for displayed divisors (None = unlimited).
# If the divisor count exceeds this limit, divisors are not shown.
DIVISORLIST_LIMIT = None              

# Aliquot sequence output settings:
ALIQUOT_SEQUENCE = True     # Show the Aliquot sequence for each number
ALIQUOTLIST_LIMIT = None    # Maximum steps to display (None = unlimited)

# OUTPUT_FILE = None or ""        # No file output
# OUTPUT_FILE = "." or "./"       # Each run creates/overwrites e.g. 123456.txt in the current directory.
# OUTPUT_FILE = "results/"        # Each run creates/overwrites e.g. results/123456.txt.
# OUTPUT_FILE = "log.txt"         # All output appended to log.txt in the current folder
# OUTPUT_FILE = "logs/run.log"    # All output appended to logs/run.log
OUTPUT_FILE = None

# ===============================================================
# End of settings.py
# ===============================================================
