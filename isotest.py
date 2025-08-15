from numclass import classify_number, TEST_FUNCTIONS_ALL, atomic_labels
from utility import analyze_divisors
from classifiers.arithmetic_divisor import is_deficient_number, is_sum_of_3_squares
from classifiers.digit_based import is_automorphic_number
from classifiers.curiosities import is_cake_number
from classifiers.prime import is_proth_prime
from user import settings

#divs, aliquot, is_prime = analyze_divisors(76)
#print("Divisors:", divs)
#print("Aliquot sum:", aliquot)
#print("Prime?", is_prime)

#print(is_deficient_number(76))

#result = classify_number(5, debug=True)
#print("Atomic results:", [c["label"] for c in result["classes"]])

#print("Automorphic number" in TEST_FUNCTIONS_ALL)
#print(TEST_FUNCTIONS_ALL.get("Automorphic number"))
#result = TEST_FUNCTIONS_ALL 
#print("is_automorphic_number(5) returned:", result, type(result))

#print(is_automorphic_number(5))

# print(settings.CATEGORIES.get("Digit-based"))
# print(settings.CATEGORIES.get("Primes & Prime-related Numbers"))
# print("Automorphic number" in atomic_labels)
# print("Prime number" in atomic_labels)
# print("print(is_sum_of_3_squares(958568, max_decomps=None)")
# print(is_sum_of_3_squares(958568, max_decomps=None))

# print(is_cake_number(313556853809))  # Should return (True, 12345)

print(is_proth_prime(41))
print(sorted(list(UNTOUCHABLE_SET))[-10:])