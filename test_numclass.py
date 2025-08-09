"""
Test suite for numclass.py

Author: Marcel M W van Dinteren <m.vandinteren1@chello.nl>
Date: 2024-07-31

usage pytest -v

"""
 
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numclass import classify_number


# List of (test_number, [classification labels]) pairs
TEST_CASES = [
    # Arithmetic and Divisor-based (23)
    (945,   ["Abundant number"]),
    (72,    ["Achilles number"]),
    (220,   ["Amicable number"]),
    (76,    ["Deficient number"]),
    (6720,  ["Highly abundant number"]),
    (36,    ["Highly composite number"]),
    (8128,  ["Perfect number"]),
    (8,     ["Perfect power"]),
    (108,   ["Powerful number"]),
    (18,    ["Practical number"]),
    (20,    ["Semiperfect number"]),
    (12496, ["Sociable number"]),
    (30,    ["Sphenic number"]),
    (10,    ["Squarefree number"]),
    (12,    ["Sublime number"]),
    (28,    ["Sum of 2 cubes"]),
    (25,    ["Sum of 2 squares"]),
    (3,     ["Sum of 3 cubes"]),
    (3,     ["Sum of 3 squares"]),
    (60,    ["Superabundant number"]),
    (120,   ["Triperfect number"]),
    (2,     ["Untouchable number"]),
    (836,   ["Weird number"]),

    # Digit-Based (19)
    (376,   ["Automorphic number"]),
    (135,   ["Disarium number"]),
    (512,   ["Dudeney number"]),
    (3,     ["Evil number"]),
    (40585, ["Factorion"]),
    (8281,  ["Happy number"]),
    (1012,  ["Harshad number"]),
    (703,   ["Kaprekar number"]),
    (196,   ["Lychrel number"]),
    (92727, ["Narcissistic number"]),
    (7,     ["Odious number"]),
    (23432, ["Palindrome"]),
    (111,   ["Palindromic Harshad number"]),
    (7777,  ["Repdigit"]),
    (31,    ["Self number"]),
    (1210,  ["Self-descriptive number"]),
    (22,    ["Smith number"]),
    (2380,  ["Sum of 2 palindromes"]),
    (5276,  ["Sum of 3 palindromes"]),

    # Fun numbers (1)
    (1337,  ["Fun number"]),

    # Integer Sequences (17)
    (4140,  ["Bell number"]),
    (107,   ["Busy Beaver number"]),
    (16127, ["Carol number"]),
    (42,    ["Catalan number"]),
    (4718593, ["Cullen number"]),
    (366,   ["Erdős-Woods number"]),
    (144,   ["Fibonacci number"]),
    (4,     ["Hamming number"]),
    (251133297, ["Keith number"]),
    (123,   ["Lucas number"]),
    (9,     ["Lucky number"]),
    (1129760415, ["Motzkin number"]),
    (16,    ["Padovan number"]),
    (70,    ["Pell number"]),
    (1729,  ["Taxicab number"]),
    (24,    ["Tribonacci number"]),
    (106,   ["Ulam number"]),

    # Mathematical Curiosities (16)
    (796,   ["Boring number"]),
    (26,    ["Cake number"]),
    (12012, ["Cyclops number"]),
    (1089,  ["Digit-Reversal constant"]),
    (26,    ["Eban number"]),
    (1,     ["Harshad in all bases"]),
    (495,   ["Kaprekar Constant (3 digit)"]),
    (6174,  ["Kaprekar Constant (4 digit)"]),
    (163,   ["Lucky number of Euler"]),
    (3435,  ["Munchausen number"]),
    (1023456789, ["Pandigital number (0-9)"]),
    (123456789,  ["Pandigital number (1-9)"]),
    (2357,  ["Smarandache Wellin number"]),
    (88,    ["Strobogrammatic number"]),
    (7,     ["Thick prime"]),
    (11,    ["Thin prime"]),
    (1260,  ["Vampire number"]),

    # Polygonal and Figurate Numbers (12)
    (142857, ["Cyclic number"]),
    (45,    ["Harshad triangular number"]),
    (66,    ["Hexagonal number"]),
    (121,   ["Lehmer number"]),
    (171,   ["Palindromic triangular number"]),
    (70,    ["Pentagonal number"]),
    (125,   ["Perfect cube"]),
    (100,   ["Perfect square"]),
    (90,    ["Pronic number"]),
    (1111,  ["Repunit"]),
    (1001452269, ["Tetrahedral number"]),
    (91,    ["Triangular number"]),

    # Prime & Prime-related Numbers (48)
    #     Atomic Primes (28)
    (113,   ["Absolute prime"]),
    (53,    ["Balanced prime"]),
    (5,     ["Catalan prime"]),
    (89,    ["Chen prime"]),
    (197,   ["Circular prime"]),
    (967,   ["Cousin prime"]),
    (13,    ["emirp"]),
    (5039,  ["Factorial prime"]),
    (65537, ["Fermat prime"]),
    (61,    ["Gaussian prime"]),
    (967,   ["Good prime"]),
    (197,   ["Keith prime"]),
    (13,    ["Left-truncatable prime"]),
    (8191,  ["Mersenne prime"]),
    (97,    ["Pierpont prime"]),
    (30029, ["Primorial prime"]),
    (41,    ["Proth prime"]),
    (127,   ["Ramanujan prime"]),
    (59,    ["Right-truncatable prime"]),
    (47,    ["Safe prime"]),
    (33,    ["Semiprime"]),
    (461,   ["Sexy prime"]),
    (23,    ["Sophie Germain prime"]),
    (37,    ["Strong prime"]),
    (59,    ["Super prime"]),
    (17,    ["Twin prime"]),
    (1093,  ["Wieferich prime"]),
    (563,   ["Wilson prime"]),
    # Intersection primes (20)
    (5,     ["Automorphic prime"]),
    (877,   ["Bell prime"]),
    (313,   ["Both-truncatable prime"]),
    (89,    ["Disarium prime"]),
    (2,     ["Factorion prime"]),
    (233,   ["Fibonacci prime"]),
    (383,   ["Happy palindromic prime"]),
    (79,    ["Happy prime"]),
    (2,     ["Harshad prime"]),
    (199,   ["Lucas prime"]),
    (223,   ["Lucky prime"]),
    (15511, ["Motzkin prime"]),
    (7,     ["Narcissistic prime"]),
    (37,    ["Padovan prime"]),
    (787,   ["Palindromic prime"]),
    (29,    ["Pell prime"]),
    (11,    ["Repunit prime"]),
    (3,     ["Triangular prime"]),
    (47,    ["Ulam prime"]),

    # Pseudoprimes and Cryptographic Numbers (7)
    (561,   ["Carmichael number"]),
    (561,   ["Euler-Jacobi pseudoprime"]), # base 2
    (121,   ["Euler-Jacobi pseudoprime"]), # base 3
    (781,   ["Euler-Jacobi pseudoprime"]), # base 5
    (325,   ["Euler-Jacobi pseudoprime"]), # base 7
    (133,   ["Euler-Jacobi pseudoprime"]), # base 11
    (85,    ["Euler-Jacobi pseudoprime"]), # base 13
    (341,   ["Fermat pseudoprime"]), # base 2
    (121,   ["Fermat pseudoprime"]), # base 3
    (781,   ["Fermat pseudoprime"]), # base 5
    (10585, ["Fermat pseudoprime"]), # base 7
    (2465,  ["Fermat pseudoprime"]), # base 11
    (244,   ["Fermat pseudoprime"]), # base 13
]

# Prepare items and IDs for pytest parametrization
TEST_ITEMS = TEST_CASES
TEST_IDS = [f"{','.join(labels)}_{n}" for n, labels in TEST_ITEMS]


@pytest.mark.parametrize("n,expected_labels", TEST_ITEMS, ids=TEST_IDS)
def test_isolated_classification(n, expected_labels):
    result = classify_number(n)
    # Extract only the string labels for easier checks
    classes = [entry["label"] for entry in result["classes"]]
    print("TEST SEES CLASSES:", classes)  # <-- put this print HERE
    # Check that each expected label is present somewhere
    for label in expected_labels:
        assert label in classes, f"{n}: expected {label} in classes, got {classes}"
