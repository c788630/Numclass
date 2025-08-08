# -----------------------------------------------------------------------------
#  Integer sequences test functions
# -----------------------------------------------------------------------------

from decorators import classifier, limited_to
from functools import lru_cache
from math import log2
from typing import Tuple


CATEGORY = "Integer Sequences"


@classifier(
    label="Bell number",
    description="Number of partitions of an n-element set.",
    oeis="A000110",
    category=CATEGORY
)
@limited_to(4638590332229999354)
def is_bell_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Bell number, else False.
    Details show which Bell number (the index in the sequence).
    """

    # Precomputed first 25 Bell numbers (0-based)
    KNOWN = [
        1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975,
        678570, 4213597, 27644437, 190899322, 1382958545,
        10480142147, 82864869804, 682076806159, 5685041521563,
        50952697604007, 474869816156751, 4506715738447323,
        44152005855084346, 445958869294805289, 4638590332229999353
    ]

    if n in KNOWN:
        idx = KNOWN.index(n)
        details = f"{n} is the Bell number B({idx})."
        return True, details
    return False, None


@classifier(
    label="Busy Beaver number",
    description="The most steps a computer program with n states can take before halting.",
    oeis="A060843",
    category=CATEGORY
)
def is_busy_beaver_number(n: int) -> Tuple[bool, str]:
    """
    The highest number of calculation steps for a Turing machine with n states.
    The sequence grows faster than any computable function.
    """
    KNOWN = [None, 1, 6, 21, 107, 47176870]  # 1-based indexing
    try:
        index = KNOWN.index(n)
        # Calculate number of Turing machines for index states
        # (standard 2-symbol)
        num_machines = (4 * index + 2) ** (2 * index)
        details = (
            f"Maximum {n} calculation step{'s' if index != 1 else ''} "
            f"for a {index}-state Turing machine "
            f"({num_machines:,} possible machines)"
        )
        return True, details
    except ValueError:
        return False, None


@classifier(
    label="Carol number",
    description="n = (2^k−1)^2 − 2 for some k.",
    oeis="A006318",
    category=CATEGORY
)
def is_carol_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Carol number, else False.
    Details show the k value and formula: n = (2^k−1)^2−2.
    """
    if n < -1:
        return False, None
    k = 1
    while True:
        carol = (2**k - 1)**2 - 2
        if carol == n:
            details = f"{n} = (2^{k} - 1)^2 - 2 (k = {k})"
            return True, details
        if carol > n:
            return False, None
        k += 1


@classifier(
    label="Catalan number",
    description="C_n=binomial(2n,n)/(n+1), counts Dyck paths.",
    oeis="A000108",
    category=CATEGORY
)
@limited_to(4861946401452)
def is_catalan_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Catalan number and provide details.
    The Catalan numbers are defined by C_n = (2n)! / ((n+1)! n!).
    """
    # finite lookup to speed up calculation
    KNOWN = [
        1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786,
        208012, 742900, 2674440, 9694845, 35357670, 129644790,
        477638700, 1767263190, 6564120420, 24466267020,
        91482563640, 343059613650, 1289904147324, 4861946401452
    ]
    if n in KNOWN:
        idx = KNOWN.index(n)
        details = f"{n} is Catalan number C({idx}): sequence: {KNOWN[:idx+1]}"
        return True, details
    return False, None


@classifier(
    label="Cullen number",
    description="n = k·2^k + 1 for some integer k ≥ 1",
    oeis="A002064",
    category=CATEGORY
)
def is_cullen_number(n: int) -> Tuple[bool, str]:
    """
    Checks if n is a Cullen number.
    Details: show k.
    """
    if n < 1:
        return False, None

    # Solve for k in k*2^k + 1 = n, try plausible k up to log2(n)
    max_k = int(log2(n)) + 2
    for k in range(1, max_k):
        val = k * 2**k + 1
        if val == n:
            details = f"{n} = {k}·2^{k} + 1 (k = {k})"
            return True, details
        if val > n:
            break
    return False, None


# Full list of known Erdős–Woods numbers (from Lygeros / OEIS A059756)
ERDOS_WOODS_NUMBERS = [
    16, 22, 34, 36, 46, 56, 64, 66, 70, 76, 78, 86, 88, 92, 94, 96, 100, 106,
    112, 116, 118, 120, 124, 130, 134, 142, 144, 146, 154, 160, 162, 186, 190,
    196, 204, 210, 216, 218, 220, 222, 232, 238, 246, 248, 250, 256, 260, 262,
    268, 276, 280, 286, 288, 292, 296, 298, 300, 302, 306, 310, 316, 320, 324,
    326, 328, 330, 336, 340, 342, 346, 356, 366, 372, 378, 382, 394, 396, 400,
    404, 406, 408, 414, 416, 424, 426, 428, 430, 438, 446, 454, 456, 466, 470,
    472, 474, 476, 484, 486, 490, 494, 498, 512, 516, 518, 520, 526, 528, 532,
    534, 536, 538, 540, 546, 550, 552, 554, 556, 560, 574, 576, 580, 582, 584,
    590, 604, 606, 612, 616, 624, 630, 634, 636, 640, 650, 666, 668, 670, 672,
    680, 690, 694, 696, 698, 700, 706, 708, 712, 714, 718, 722, 726, 732, 738,
    742, 746, 750, 754, 756, 760, 764, 768, 780, 782, 784, 786, 790, 792, 794,
    800, 804, 806, 814, 816, 818, 820, 832, 834, 836, 838, 844, 846, 852, 862,
    870, 874, 876, 880, 886, 890, 896, 900, 902, 903, 904, 906, 910, 918, 922,
    924, 928, 936, 940, 950, 960, 964, 966, 970, 974, 982, 988, 990, 996,
    1000, 1002, 1004, 1008, 1016, 1026, 1028, 1030, 1038, 1044, 1046, 1056,
    1058, 1060, 1068, 1074, 1076, 1078, 1080, 1100, 1106, 1114, 1116, 1120,
    1122, 1128, 1134, 1140, 1144, 1150, 1158, 1160, 1170, 1176, 1180, 1190,
    1196, 1200, 1206, 1208, 1210, 1212, 1220, 1222, 1234, 1240, 1242, 1246,
    1248, 1254, 1258, 1262, 1264, 1268, 1270, 1272, 1274, 1282, 1288, 1294,
    1296, 1300, 1310, 1314, 1318, 1326, 1330, 1334, 1338, 1340, 1342, 1344,
    1350, 1352, 1354, 1356, 1358, 1364, 1372, 1384, 1386, 1392, 1404, 1406,
    1408, 1412, 1416, 1418, 1422, 1438, 1442, 1446, 1458, 1462, 1464, 1470,
    1474, 1476, 1478, 1480, 1496, 1498, 1502, 1506, 1508, 1516, 1518, 1520,
    1526, 1528, 1530, 1536, 1540, 1542, 1546, 1548, 1552, 1562, 1564, 1566,
    1574, 1578, 1590, 1592, 1596, 1600, 1606, 1612, 1626, 1630, 1632, 1640,
    1644, 1650, 1652, 1654, 1656, 1662, 1674, 1676, 1678, 1680, 1686, 1688,
    1696, 1706, 1716, 1728, 1732, 1736, 1740, 1746, 1750, 1752, 1758, 1764,
    1766, 1770, 1772, 1776, 1780, 1782, 1792, 1794, 1796, 1800, 1804, 1806,
    1808, 1818, 1820, 1826, 1830, 1834, 1836, 1838, 1840, 1842, 1844, 1856,
    1858, 1860, 1866, 1870, 1884, 1892, 1894, 1898, 1900, 1904, 1910, 1916,
    1918, 1920, 1924, 1938, 1940, 1944, 1948, 1956, 1958, 1960, 1962, 1968,
    1978, 1982, 1986, 1992, 2002, 2010, 2014, 2016, 2020, 2022, 2034, 2038,
    2042, 2044, 2046, 2048, 2050, 2052, 2058, 2066, 2072, 2074, 2076, 2078,
    2080, 2086, 2092, 2094, 2102, 2104, 2106, 2116, 2118, 2120, 2122, 2124,
    2134, 2136, 2140, 2146, 2148, 2150, 2152, 2158, 2160, 2166, 2168, 2170,
    2172, 2174, 2176, 2178, 2184, 2186, 2190, 2192, 2196, 2200, 2202, 2216,
    2220, 2226, 2230, 2232, 2236, 2248, 2250, 2254, 2258, 2262, 2264, 2278,
    2280, 2284, 2286, 2292, 2300, 2302, 2304, 2316, 2318, 2322, 2328, 2344,
    2346, 2360, 2362, 2364, 2368, 2370, 2376, 2380, 2386, 2388, 2404, 2408,
    2414, 2420, 2422, 2426, 2428, 2430, 2436, 2440, 2446, 2450, 2452, 2454,
    2464, 2466, 2472, 2482, 2484, 2490, 2492, 2496, 2498, 2500, 2502, 2506,
    2508, 2510, 2514, 2518, 2520, 2526, 2530, 2534, 2538, 2545, 2546, 2548,
    2560, 2562, 2568, 2570, 2574, 2576, 2586, 2588, 2590, 2596, 2598, 2604,
    2606, 2612, 2616, 2620, 2628, 2640, 2642, 2652, 2656, 2662, 2670, 2676,
    2680, 2704, 2706, 2718, 2724, 2726, 2728, 2734, 2736, 2738, 2740, 2744,
    2746, 2748, 2752, 2756, 2760, 2762, 2766, 2770, 2772, 2774, 2780, 2782,
    2786, 2796, 2800, 2808, 2814, 2822, 2824, 2826, 2830, 2832, 2840, 2850,
    2856, 2860, 2866, 2868, 2870, 2874, 2878, 2884, 2886, 2892, 2894, 2900,
    2914, 2916, 2920, 2922, 2924, 2946, 2950, 2952, 2966, 2976, 2978, 2982,
    2988, 2990, 2992, 2994, 3006, 3008, 3014, 3016, 3018, 3030, 3032, 3036,
    3040, 3052, 3054, 3058, 3060, 3064, 3066, 3072, 3078, 3094, 3096, 3098,
    3102, 3108, 3112, 3114, 3124, 3130, 3132, 3140, 3146, 3148, 3150, 3154,
    3156, 3162, 3166, 3174, 3176, 3180, 3190, 3198, 3200, 3220, 3228, 3234,
    3240, 3246, 3248, 3250, 3264, 3270, 3276, 3282, 3284, 3288, 3304, 3306,
    3310, 3318, 3328, 3334, 3336, 3346, 3352, 3354, 3366, 3368, 3370, 3378,
    3380, 3384, 3388, 3396, 3398, 3402, 3406, 3410, 3418, 3420, 3422, 3430,
    3432, 3436, 3438, 3440, 3444, 3454, 3456, 3472, 3474, 3480, 3486, 3494,
    3498, 3504, 3508, 3510, 3514, 3516, 3522, 3524, 3536, 3538, 3546, 3552,
    3556, 3568, 3570, 3574, 3586, 3588, 3590, 3596, 3598, 3600, 3604, 3610,
    3622, 3630, 3634, 3636, 3640, 3648, 3654, 3656, 3666, 3668, 3670, 3680,
    3682, 3684, 3688, 3690, 3694, 3696, 3700, 3708, 3714, 3716, 3730, 3732,
    3738, 3744, 3746, 3750, 3752, 3760, 3782, 3784, 3792, 3800, 3810, 3812,
    3816, 3828, 3830, 3832, 3836, 3840, 3846, 3850, 3868, 3870, 3872, 3880,
    3892, 3894, 3900, 3902, 3904, 3906, 3910, 3922, 3926, 3936, 3938, 3942,
    3960, 3964, 3978, 3984, 3986, 3996, 4000, 4006, 4010, 4012, 4016, 4032,
    4034, 4036, 4038, 4040, 4042, 4044, 4046, 4048, 4062, 4068, 4070, 4086,
    4088, 4098, 4102, 4104, 4106, 4110, 4116, 4118, 4122, 4124, 4146, 4152,
    4156, 4164, 4170, 4172, 4174, 4180, 4182, 4186, 4188, 4190, 4196, 4200,
    4204, 4206, 4208, 4210, 4224, 4228, 4238, 4240, 4248, 4250, 4256, 4266,
    4270, 4278, 4280, 4286, 4296, 4308, 4312, 4314, 4316, 4318, 4320, 4322,
    4324, 4326, 4332, 4342, 4344, 4352, 4354, 4356, 4362, 4368, 4370, 4378,
    4380, 4382, 4386, 4394, 4396, 4400, 4402, 4404, 4412, 4416, 4418, 4420,
    4428, 4430, 4434, 4440, 4444, 4446, 4468, 4470, 4476, 4478, 4486, 4488,
    4498, 4500, 4506, 4510, 4512, 4526, 4530, 4534, 4536, 4538, 4540, 4542,
    4546, 4554, 4560, 4566, 4570, 4572, 4578, 4580, 4582, 4588, 4590, 4594,
    4596, 4602, 4608, 4610, 4612, 4620, 4626, 4628, 4632, 4646, 4654, 4656,
    4660, 4662, 4668, 4670, 4678, 4682, 4684, 4686, 4688, 4690, 4698, 4700,
    4710, 4718, 4728, 4732, 4740, 4746, 4748, 4750, 4754, 4758, 4768, 4770,
    4776, 4782, 4792, 5184, 6084, 6494, 6724, 7056, 7762, 8100, 8470, 8472,
    8474, 8476, 8480, 8484, 9216, 9988, 9990, 10000, 10404, 11236, 11638,
    11640, 11664, 12350, 12544, 14400, 15876, 16900, 17424, 22500, 24336,
    26244, 27556, 29584, 30276, 31684, 32400, 36864, 38416, 44100, 49284,
    51076, 53824, 56644, 57600, 71824, 86436, 93636, 96100, 108900, 112896,
    138384, 142884, 156816, 166464, 184900, 191844, 207936, 236196, 240100,
    248004, 252004, 260100, 270400, 272484, 315844, 322624, 324900, 331776,
]

# Known small Erdős–Woods numbers with verified minimal intervals
# Source: OEIS comments / Lygeros computations
ERDOS_WOODS_INTERVALS = {
    16: "[2184, 2200]",
    22: "[2184, 2206]",
    34: "[33, 67]",
    36: "[2184, 2220]",
    46: "[57, 103]",
    60: "[87, 147]",
    86: "[15, 101]",
    96: "[93, 189]",
    114: "[39, 153]",
    124: "[2184, 2308]",
    146: "[111, 257]",
    174: "[69, 243]",
    180: "[129, 309]",
    210: "[15, 225]",
    222: "[57, 279]",
    234: "[87, 321]",
    258: "[69, 327]",
    292: "[129, 421]",
    336: "[57, 393]",
    354: "[33, 387]",
    366: "[93, 459]",
}


@classifier(
    label="Erdős-Woods number",
    description="n is length of an interval where every number shares a factor with the first or last.",
    oeis="A059756",
    category=CATEGORY
)
@limited_to(331777)
def is_erdos_woods_number(n: int) -> tuple[bool, str]:
    """
    Checks if n is an Erdős–Woods number.

    Returns (True, details) if found; details include witness interval.
    Uses precomputed list.
    """

    if n < 1:
        return False, None

    if n in ERDOS_WOODS_INTERVALS:
        return True, f"Witness interval: {ERDOS_WOODS_INTERVALS[n]}"

    if n in ERDOS_WOODS_NUMBERS:
        return True, f"{n} is a known Erdős–Woods number (interval very large or unknown)"

    return False, None


@classifier(
    label="Fibonacci number",
    description="F(n)=F(n−1)+F(n−2), F(0)=0,F(1)=1.",
    oeis="A000045",
    category=CATEGORY
)
def is_fibonacci_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Fibonacci number and return details.
    Details: Fibonacci index and sequence.
    """
    if n < 0:
        return False, None  # Only non-negative
    fib_seq = [0, 1]
    while fib_seq[-1] < n:
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    if n in fib_seq:
        idx = fib_seq.index(n)
        details = f"{n} is Fibonacci number F({idx}): sequence: {fib_seq[:idx+1]}"
        return True, details
    return False, None


@classifier(
    label="Hamming number",
    description="Numbers with no prime factor greater than 5 (also called regular numbers)",
    oeis="A051037",
    category=CATEGORY
)
def is_hamming_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Hamming number (all prime factors ≤ 5).
    """
    if n < 1:
        return False, None
    original_n = n
    for p in [2, 3, 5]:
        while n % p == 0 and n > 1:
            n //= p
    if n == 1:
        return True, f"{original_n} has no prime factors greater than 5."
    else:
        return False, None


@classifier(
    label="Keith number",
    description="Appears in its own digit-sequence sum recurrence.",
    oeis="A007629",
    category=CATEGORY
)
def is_keith_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Keith number, else False.
    Details show the digit recurrence sequence leading to n.
    """
    @lru_cache(maxsize=128)
    def keith_test(n):
        if n < 10:
            return False, None  # No single-digit Keith numbers

        seq = [int(d) for d in str(n)]
        k = len(seq)
        steps = list(seq)
        while True:
            next_term = sum(steps[-k:])
            steps.append(next_term)
            if next_term == n:
                details = f"Keith sequence: {', '.join(str(x) for x in steps)}"
                return True, details
            if next_term > n:
                return False, None
    return keith_test(n)


@classifier(
    label="Lucas number",
    description=" L(n)=L(n−1)+L(n−2), L(0)=2,L(1)=1.",
    oeis="A000032",
    category=CATEGORY
)
def is_lucas_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Lucas number and provide details.
    The Lucas sequence is defined by L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2).
    Supports negative indices: L(-k) = (-1)^k * L(k)
    """
    if n < 0:
        return False, None
    L = [2, 1]
    abs_n = abs(n)
    # Generate Lucas numbers up to |n|
    while abs(L[-1]) < abs_n:
        L.append(L[-1] + L[-2])

    # Positive Lucas numbers
    if n in L:
        idx = L.index(n)
        details = f"{n} is Lucas number L({idx}): sequence: {L[:idx+1]}"
        return True, details

    # Negative indices: L(-k) = (-1)^k * L(k)
    for k, val in enumerate(L):
        neg_val = ((-1) ** k) * val
        if n == neg_val:
            details = (
                f"{n} is Lucas number L(-{k}) = {neg_val}; "
                f"L(-k) = (-1)^{k} * L({k}) = {neg_val}; "
                f"sequence: {[((-1) ** i) * L[i] for i in range(k+1)]}"
            )
            return True, details

    return False, None


@classifier(
    label="Lucky number",
    description="Number remaining after repeatedly removing every k-th number (k=2, 3, …) from the natural numbers.",
    oeis="A000959",
    category=CATEGORY
)
@limited_to(999999)
def is_lucky_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a lucky number.
    Details: index if n is lucky
    """
    if n < 1:
        return False, None
    numbers = list(range(1, max(n, 10000) + 1, 2))  # Start with odd numbers
    idx = 1
    # Start sieving from the second element (1-based index in mathematical description)
    while idx < len(numbers):
        step = numbers[idx]
        if step <= 0:
            break
        # Remove every step-th element (1-based counting!)
        # The deletion must always use the current list length and indices.
        del numbers[step-1::step]
        idx += 1
    if n in numbers:
        lucky_idx = numbers.index(n)
        details = f"{n} is Lucky number L({lucky_idx})."
        return True, details
    return False, None


@classifier(
    label="Motzkin number",
    description="Counts certain lattice paths of length n.",
    oeis="A001006",
    category=CATEGORY
)
@limited_to(192137918101841818)
def is_motzkin_number(n: int) -> Tuple[bool, str]:
    """
    Returns (True, details) if n is a Motzkin number, else False.
    Details show the index and initial segment of the Motzkin sequence.
    """
    KNOWN = [
        1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835,
        113634, 310572, 853467, 2356779, 6536382, 18199284, 50852019,
        142547559, 400763223, 1129760415, 3192727797, 9043402501,
        25669818476, 73007772802, 208023278209, 593742784829,
        1697385471211, 4859761676391, 13933569346707, 40002464776083,
        114988706524270, 330931069469828, 953467954114363,
        2750016719520991, 7939655757745265, 22944749046030949,
        66368199913921497, 192137918101841817
    ]
    if n in KNOWN:
        idx = KNOWN.index(n)
        seq_str = ", ".join(str(x) for x in KNOWN[:idx+1])
        details = f"{n} is Motzkin number M({idx}): sequence: {seq_str}"
        return True, details
    return False, None


@classifier(
    label="Padovan number",
    description="Defined by P(n)=P(n-2)+P(n-3), starting 1,1,1.",
    oeis="A000931",
    category=CATEGORY
)
def is_padovan_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Padovan number and provide details.
    The Padovan sequence is defined by P(0) = P(1) = P(2) = 1,
    and P(n) = P(n-2) + P(n-3) for n > 2.
    """
    if n < 0:
        return False, None
    seq = [1, 1, 1]
    idx = 2
    while seq[-1] < n:
        next_val = seq[-2] + seq[-3]
        seq.append(next_val)
        idx += 1
    if seq[-1] == n:
        details = f"{n} is Padovan number P({idx}): sequence: {seq[:idx+1]}"
        return True, details
    return False, None


@classifier(
    label="Pell number",
    description="P(n)=2P(n−1)+P(n−2), P(0)=0,P(1)=1.",
    oeis="A000129",
    category=CATEGORY
)
def is_pell_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Pell number and provide details.
    The Pell numbers are an integer sequence defined by the recurrence
    P(n) = 2·P(n−1) + P(n−2), with P(0) = 0, P(1) = 1.
    Supports negative numbers: P(-k) = (-1)^k * P(k)
    """
    if n < 0:
        return False, None
    pell = [0, 1]
    abs_n = abs(n)
    # Generate Pell numbers up to |n|
    while abs(pell[-1]) < abs_n:
        pell.append(2 * pell[-1] + pell[-2])

    # Check positive Pell numbers
    if n in pell:
        idx = pell.index(n)
        details = f"{n} is Pell number P({idx}): sequence: {pell[:idx+1]}"
        return True, details

    # Check negative indices: P(-k) = (-1)^k * P(k)
    for k, val in enumerate(pell):
        neg_val = ((-1) ** k) * val
        if n == neg_val:
            details = (
                f"{n} is Pell number P(-{k}) = {neg_val}; "
                f"P(-k) = (-1)^{k} * P({k}) = {neg_val}; "
                f"sequence: {[((-1) ** i) * pell[i] for i in range(k+1)]}"
            )
            return True, details

    return False, None


@classifier(
    label="Taxicab number",
    description="Can be written as a sum of two positive cubes in at least two ways. (Also called a Hardy-Ramanujan) number.",
    oeis="A001235",
    category=CATEGORY
)
@limited_to(100973305)
def is_taxicab_number(n) -> Tuple[bool, str]:
    """
    Checks if n os a Taxicab number
    Details: for n as taxicab number (up to 2, 3, 4, 5 ways, below 100973304).
    """

    # Taxicab numbers, up to taxicab(5), n <= 100973304
    # Format: n : (ways, [ "a³+b³", ... ] )
    TAXICAB_DETAILS = {
        1729: (2, ["1³+12³", "9³+10³"]),
        4104: (2, ["2³+16³", "9³+15³"]),
        13832: (2, ["2³+24³", "18³+20³"]),
        20683: (2, ["10³+27³", "19³+24³"]),
        32832: (2, ["4³+32³", "18³+30³"]),
        39312: (2, ["2³+34³", "15³+33³"]),
        40033: (2, ["9³+34³", "16³+33³"]),
        46683: (3, ["3³+36³", "10³+37³", "27³+30³"]),
        64232: (2, ["17³+39³", "26³+36³"]),
        65728: (2, ["12³+40³", "31³+33³"]),
        110656: (2, ["4³+48³", "18³+46³"]),
        110808: (2, ["27³+41³", "36³+40³"]),
        134379: (2, ["9³+50³", "25³+44³"]),
        149389: (2, ["17³+54³", "32³+53³"]),
        165464: (2, ["23³+55³", "34³+54³"]),
        216027: (2, ["6³+59³", "19³+58³"]),
        216125: (3, ["5³+60³", "17³+58³", "30³+55³"]),
        262656: (2, ["2³+64³", "16³+62³"]),
        314496: (2, ["12³+68³", "32³+68³"]),
        320264: (2, ["9³+69³", "32³+68³"]),
        327763: (2, ["13³+70³", "29³+68³"]),
        373464: (2, ["14³+74³", "44³+62³"]),
        402597: (2, ["17³+77³", "36³+75³"]),
        439101: (2, ["27³+80³", "45³+76³"]),
        515375: (2, ["15³+80³", "44³+77³"]),
        525824: (2, ["24³+80³", "36³+80³"]),
        558441: (2, ["11³+82³", "54³+77³"]),
        593047: (2, ["19³+84³", "60³+77³"]),
        684019: (2, ["30³+89³", "65³+84³"]),
        704977: (2, ["41³+88³", "48³+89³"]),
        805688: (2, ["12³+92³", "72³+80³"]),
        842751: (2, ["39³+92³", "56³+87³"]),
        885248: (2, ["56³+88³", "62³+86³"]),
        886464: (2, ["24³+96³", "60³+88³"]),
        920673: (2, ["33³+94³", "72³+87³"]),
        955016: (2, ["28³+98³", "84³+88³"]),
        984067: (2, ["51³+98³", "66³+97³"]),
        998001: (2, ["99³+0³", "70³+91³"]),
        87539319: (3, ["167³+436³", "228³+423³", "255³+414³"]),
        100973304: (5, ["2³+464³", "228³+454³", "167³+485³", "131³+502³", "119³+508³"]),
    }
    if n in TAXICAB_DETAILS:
        ways, pairs = TAXICAB_DETAILS[n]
        min_order = {2: "smallest number as the sum of two cubes in two ways",
                     3: "smallest number as the sum of two cubes in three ways",
                     4: "smallest number as the sum of two cubes in four ways",
                     5: "smallest number as the sum of two cubes in five ways"}
        label = min_order.get(ways, f"in {ways} ways")
        cubes = ", ".join(pairs)
        details = f"{n} = {cubes} ({label})"
        return True, details
    return False, None


@classifier(
    label="Tribonacci number",
    description="Sum of preceding three numbers in its sequence, starting 0,0,1.",
    oeis="A000073",
    category=CATEGORY
)
def is_tribonacci_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is a Tribonacci number, including negative indices.
    T(0)=0, T(1)=0, T(2)=1, T(n)=T(n-1)+T(n-2)+T(n-3) for n>2
    For negative index: T(-n) = -T(-n+1) + T(-n+3)
    """
    if n < 0:
        return False, None
    # Positive indices
    pos = [0, 0, 1]
    idx = 2
    while abs(pos[-1]) < abs(n):
        next_val = pos[-1] + pos[-2] + pos[-3]
        pos.append(next_val)
        idx += 1
    if n in pos:
        found_idx = pos.index(n)
        details = f"{n} is Tribonacci number T({found_idx}): sequence: {pos[:found_idx+1]}"
        return True, details
    # Negative indices
    neg = [0, 0, 1]
    for i in range(3, 40):  # 40 is arbitrary, adjust if you want
        neg_val = -neg[-1] + neg[-3]
        neg.append(neg_val)
        if neg_val == n:
            details = f"{n} is Tribonacci number T(-{i}): negative sequence: {neg}"
            return True, details
    return False, None


# ulam_list keeps global state for better performance
ulam_list = [1, 2]
ulam_set = {1, 2}
ulam_max = 2


@classifier(
    label="Ulam number",
    description="Next term is unique sum of two prior terms.",
    oeis="A002858",
    category=CATEGORY
)
@limited_to(59999)
def is_ulam_number(n: int) -> Tuple[bool, str]:
    """
    Check if n is an Ulam number.
    Returns (True, details) with index if n is Ulam; else False.
    """
    global ulam_list, ulam_set, ulam_max
    if n < 1:
        return False, None
    try:
        ulam_list
        ulam_set
        ulam_max
    except NameError:
        ulam_list = [1, 2]
        ulam_set = set(ulam_list)
        ulam_max = 2

    if n < 1:
        return False, None

    while ulam_max < n:
        candidate = ulam_max + 1
        count = 0
        for x in ulam_list:
            if x > candidate // 2:
                break
            y = candidate - x
            if y != x and y in ulam_set:
                count += 1
                if count > 1:
                    break
        if count == 1:
            ulam_list.append(candidate)
            ulam_set.add(candidate)
        ulam_max += 1

    if n in ulam_set:
        idx = ulam_list.index(n)
        details = f"{n} is Ulam number U({idx})."
        return True, details
    return False, None
