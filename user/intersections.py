# =============================================================================
# intersection_rules.py — User-customizable intersection rules for numclass
#
# This file lets you define "intersection" classifications, where a number
# that meets several atomic class conditions is recognized with a combined label.
#
# Example: If a number is both a "Prime number" and a "Palindrome",
#          the rule ("Prime number", "Palindrome") → "Palindromic prime"
#
# HOW TO USE:
#   - Each intersection rule is a tuple: (tuple of atomic labels, intersection label)
#   - You can list the rules in any order; the code always applies the longest (most specific)
#     intersections first, so triple intersections will take priority over double, etc.
#   - Atomic labels must exactly match the labels used elsewhere in numclass.
#   - Intersection labels should be unique and descriptive (not identical to atomic).
#   - You may add, remove, or comment out rules as you like.
#
# EXAMPLES:
#   - See current implementation below as example.
#
# NOTE:
#   - If you remove a rule here, the corresponding intersection will not be used.
#   - Be careful not to create ambiguous overlaps.
#   - The file is read at program startup; restart numclass to reload changes.
#
# =============================================================================

INTERSECTION_RULES = [
    (("Prime number", "Automorphic number"), "Automorphic prime"),
    (("Prime number", "Cyclops number"), "Cyclops prime"),
    (("Prime number", "Bell number"), "Bell prime"),
    (("Prime number", "Disarium number"), "Disarium prime"),
    (("Prime number", "Factorion"), "Factorion prime"),
    (("Prime number", "Fibonacci number"), "Fibonacci prime"),
    (("Prime number", "Happy number"), "Happy prime"),
    (("Prime number", "Harshad number"), "Harshad prime"),
    (("Prime number", "Lucas number"), "Lucas prime"),
    (("Prime number", "Lucky number"), "Lucky prime"),
    (("Prime number", "Motzkin number"), "Motzkin prime"),
    (("Prime number", "Narcissistic number"), "Narcissistic prime"),
    (("Prime number", "Padovan number"), "Padovan prime"),
    (("Prime number", "Palindrome"), "Palindromic prime"),
    (("Prime number", "Pell number"), "Pell prime"),
    (("Prime number", "Repunit"), "Repunit prime"),
    (("Prime number", "Triangular number"), "Triangular prime"),
    (("Prime number", "Ulam number"), "Ulam prime"),
    # Non-prime intersections:
    (("Palindrome", "Harshad number"), "Palindromic Harshad number"),
    (("Triangular number", "Palindrome"), "Palindromic triangular number"),
    (("Triangular number", "Harshad number"), "Harshad triangular number"),
    # Triple intersections:
    (("Prime number", "Happy number", "Palindrome"), "Happy palindromic prime"),
    (("Left-truncatable prime", "Right-truncatable prime"), "Both-truncatable prime"),
]