# -----------------------------------------------------------------------------
#  pseudoprime_crypto.py
#  Pseudoprimes and cryptographic number test functions
# -----------------------------------------------------------------------------

from __future__ import annotations

from math import gcd

from sympy import integer_nthroot, isprime, jacobi_symbol

from numclass.context import NumCtx
from numclass.fmt import abbr_int_fast
from numclass.registry import classifier
from numclass.utility import build_ctx

CATEGORY = "Pseudoprimes and Cryptographic Numbers"
BASES = (2, 3, 5, 7, 11, 13)


def _is_perfect_power(n: int) -> tuple[bool, int | None, int | None]:
    """
    Bigint-safe: no floats. Uses integer n-th roots; short-circuits powers of two.
    Returns (True, base, exp) if n is a perfect power with base>1, exp>1;
    else (False, None, None).
    """
    if n <= 1:
        return False, None, None

    # --- quick path: powers of two (fast bit trick) -------------------------
    # If n is a power of two, bit_count()==1 and exponent is bit_length-1
    if n & (n - 1) == 0:
        k = n.bit_length() - 1
        if k > 1:
            return True, 2, k

    a = n  # positive

    # --- iterate only PRIME exponents k (classic trick) ---------------------
    def primes_upto(L: int):
        if L < 2:
            return
        sieve = bytearray(b"\x01") * (L + 1)
        sieve[0:2] = b"\x00\x00"
        p = 2
        while p * p <= L:
            if sieve[p]:
                step = p
                sieve[p * p:L + 1:step] = b"\x00" * (((L - p * p) // step) + 1)
            p += 1
        for q in range(2, L + 1):
            if sieve[q]:
                yield q

    # largest possible exponent k satisfies 2^k <= n → k_max = floor(log2(n))
    k_max = a.bit_length() - 1  # exact, integer, no floats

    for k in primes_upto(k_max):
        # If you ever want negative bases, only allow odd k here.
        r, exact = integer_nthroot(a, k)  # r = floor(a^(1/k)), exact if r**k == a
        if exact and r > 1:
            return True, int(r), k

    return False, None, None


@classifier(
    label="Blum integer",
    description="Semiprime n = p·q with distinct primes p ≡ q ≡ 3 (mod 4).",
    oeis="A016105",
    category=CATEGORY,
)
def is_blum_integer(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n <= 0:
        return False, None
    ctx = ctx or build_ctx(abs(n))

    # need exactly two distinct prime factors, each to the first power (squarefree semiprime)
    if len(ctx.fac) != 2 or any(e != 1 for e in ctx.fac.values()):
        return False, None

    (p, e1), (q, e2) = sorted(ctx.fac.items())
    if (p % 4 == 3) and (q % 4 == 3):
        return True, f"n = {p}×{q}; {p}≡3 (mod 4), {q}≡3 (mod 4)"
    return False, None


@classifier(
    label="Carmichael number",
    description="Composite numbers passing Fermat primality tests for multiple bases.",
    oeis="A002997",
    category=CATEGORY,
    limit=9999999
)
def is_carmichael_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    """
    Returns (True, details) if n is a Carmichael number, else (False, None).

    Korselt's criterion (1899):
      • n is composite
      • n is square-free
      • For every prime p | n, (p - 1) | (n - 1)
    """
    # Must be composite and ≥ 3; all Carmichael numbers are odd
    if n < 3 or isprime(n) or (n % 2 == 0):
        return False, None

    factors = (ctx or build_ctx(abs(n))).fac

    # Square-free check
    if any(exp != 1 for exp in factors.values()):
        return False, None

    # (Optional speed tweak) Classic theorem: Carmichael numbers have ≥ 3 prime factors
    if len(factors) < 3:
        return False, None

    # Build details; use bullet style and per-prime divisibility checks
    bullet = "•"
    pf_list = " * ".join(str(p) for p in sorted(factors))
    lines = [f"Prime factors: {pf_list}"]

    ok = True
    for p in sorted(factors):
        r = (n - 1) % (p - 1)
        cond = (r == 0)
        lines.append(f"             {bullet} (n-1) % (p-1) = ({n}-1) % ({p}-1) = {r} {'✓' if cond else '✗'}")
        if not cond:
            ok = False
            break

    if not ok:
        return False, None

    lines.append("             All conditions satisfied (composite, square-free, and (p−1)|(n−1) for all p).")
    return True, "\n".join(lines)


@classifier(
    label="Cunningham number",
    description="Number of the form a^b ± 1 with a, b ≥ 2.",
    oeis="A080262",
    category=CATEGORY,
)
def is_cunningham_number(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    if n <= 2:
        return False, None

    # delta = +1 → n = a^b + 1
    # delta = -1 → n = a^b − 1
    for delta, sign_label in ((1, "+"), (-1, "−")):
        m = n - delta       # so that n = m + delta, with m = a^b
        if m <= 3:
            continue

        is_pp, a, b = _is_perfect_power(m)
        if is_pp and a >= 2 and b >= 2:
            details = (
                f"{abbr_int_fast(n)} is a Cunningham number: "
                f"{abbr_int_fast(n)} = {a}^{b} {sign_label} 1 "
                f"with a={a}, b={b}."
            )
            return True, details

    return False, None


@classifier(
    label="Euler-Jacobi pseudoprime",
    description="Composite n where a^((n-1)//2) ≡ Jacobi(a,n) mod n for at least one base in {2,3,5,7,11,13}.",
    oeis="A047713",
    category=CATEGORY,
    limit=10**1000 - 1
)
def _is_euler_jacobi_pseudoprime(n: int, bases=BASES):
    """
    Euler–Jacobi pseudoprime test:
    Composite n where a^((n-1)//2) ≡ Jacobi(a, n) mod n for at least one base in bases.
    """
    if n < 3 or n % 2 == 0 or isprime(n):
        return False, None

    passing = []
    exp = (n - 1) // 2
    for a in bases:
        if gcd(a, n) != 1:
            continue
        jac = jacobi_symbol(a, n)
        jac_mod = jac % n
        if jac == -1:
            jac_mod = n - 1
        res = pow(a, exp, n)
        if res == jac_mod:
            passing.append(f"{a}^{exp} ≡ {res} (mod {n}), Jacobi({a},{n})={jac}")

    if passing:
        return True, "; ".join(passing)
    return False, None


@classifier(
    label="Fermat pseudoprime",
    description="Composite n such that a^(n-1) ≡ 1 mod n for at least one base in {2,3,5,7,11,13}.",
    oeis="A001567",  # base 2 only; multi-base sets are separate sequences
    category=CATEGORY,
    limit=10**1000 - 1
)
def _is_fermat_pseudoprime(n: int, bases=BASES):
    """
    Fermat pseudoprime test:
    Composite n such that a^(n-1) ≡ 1 mod n for at least one base in bases.
    """
    if n < 3 or isprime(n):
        return False, None

    passing = []
    for a in bases:
        if gcd(a, n) != 1:
            continue
        res = pow(a, n-1, n)
        if res == 1:
            passing.append(f"Base:{a} {a}^{n-1} ≡ 1 (mod {n})")

    if passing:
        return True, "; ".join(passing)
    return False, None


@classifier(
    label="Lucas-Carmichael number",
    description="Odd squarefree composite n with p+1 | n+1 for all primes p|n.",
    oeis="A006972",
    category=CATEGORY,
)
def is_lucas_carmichael(n: int, ctx: NumCtx | None = None) -> tuple[bool, str | None]:
    # Must be odd, composite, and squarefree
    if n <= 1 or (n & 1) == 0:
        return False, None
    ctx = ctx or build_ctx(n)
    # composite & squarefree: at least two prime factors and all exponents == 1
    if len(ctx.fac) < 2 or any(e != 1 for e in ctx.fac.values()):
        return False, None

    primes = sorted(ctx.fac.keys())
    if all((n + 1) % (p + 1) == 0 for p in primes):
        parts = ", ".join(f"{p+1} | {n+1}" for p in primes)
        return True, f"primes(n)={'×'.join(map(str, primes))}; {parts}"
    return False, None


# Store RSA challenge numbers and their factorizations as strings.
# You can extend this dict whenever you like (RSA-150, 155, 160, …).
RSA_CHALLENGES: dict[str, dict[str, str]] = {
    "RSA-100": {
        "n": "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139",
        "p": "37975227936943673922808872755445627854565536638199",
        "q": "40094690950920881030683735292761468389214899724061",
    },
    "RSA-110": {
        "n": "35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667",
        "p": "6122421090493547576937037317561418841225758554253106999",
        "q": "5846418214406154678836553182979162384198610505601062333",
    },
    "RSA-120": {
        "n": (
            "227010481295437363334259960947493668895875336466084780038173258247009162"
            "675779735389791151574049166747880487470296548479"
        ),
        "p": "327414555693498015751146303749141488063642403240171463406883",
        "q": "693342667110830181197325401899700641361965863127336680673013",
    },
    "RSA-129": {
        "n": (
            "114381625757888867669235779976146612010218296721242362562561842935706935"
            "245733897830597123563958705058989075147599290026879543541"
        ),
        "p": "3490529510847650949147849619903898133417764638493387843990820577",
        "q": "32769132993266709549961988190834461413177642967992942539798288533",
    },
    "RSA-130": {
        "n": (
            "180708208868740480595165616440590556627810251676940134917012702145005666"
            "2540244048387341127590812303371781887966563182013214880557"
        ),
        "p": "39685999459597454290161126162883786067576449112810064832555157243",
        "q": "45534498646735972188403686897274408864356301263205069600999044599",
    },
    "RSA-140": {
        "n": (
            "212902463182587575474978820162715174978067039632772162782333832153819499"
            "84056495911366573853021918316783107387995317230889569230873441936471"
        ),
        "p": "3398717423028438554530123627613875835633986495969597423490929302771479",
        "q": "6264200187401285096151654948264442219302037178623509019111660653946049",
    },
    "RSA-150": {
        "n": (
            "155089812478348440509606754370011861770654545830995430655466945774312632"
            "70346346595436333502757772902539145399678741402700350163177218684089079"
            "5964683"
        ),
        "p": "348009867102283695483970451047593424831012817350385456889559637548278410717",
        "q": "445647744903640741533241125787086176005442536297766153493419724532460296199",
    },
    "RSA-155": {
        "n": (
            "109417386415705274218097073220403576120037329454492059909138421314763499"
            "84288934784717997257891267332497625752899781833797076537244027146743531"
            "593354333897"
        ),
        "p": "102639592829741105772054196573991675900716567808038066803341933521790711307779",
        "q": "106603488380168454820927220360012878679207958575989291522270608237193062808643",
    },
    "RSA-160": {
        "n": (
            "215274110271888970189601520131282542925777358884567598017049767677813314"
            "52188591356730110597734910596024979071115852143020793146652028401406199"
            "46994927570407753"
        ),
        "p": "45427892858481394071686190649738831656137145778469793250959984709250004157335359",
        "q": "47388090603832016196633832303788951973268922921040957944741354648812028493909367",
    },
    "RSA-576": {
        "n": (
            "188198812920607963838697239461650439807163563379417382700763356422988859"
            "71523466548531906060650474304531738801130339671619969232120573403187955"
            "0656996221305168759307650257059"
        ),
        "p": "398075086424064937397125500550386491199064362342526708406385189575946388957261768583317",
        "q": "472772146107435302536223071973048224632914695302097116459852171130520711256363590397527",
    },
    "RSA-640": {
        "n": (
            "310741824049004372135075003588856793003734602284272754572016194614983"
            "2898257138539397830247483191830625012427178427726584003348283198858598"
            "401429704901835796076605329379963587409280333006993"
        ),
        "p": (
            "296690933370964657135064023675423744972676847227507517934632702678279"
            "06355678914404494229195865540380177240578068455549772234298784552925"
        ),
        "q": (
            "10481763972441665315532284649274585382797333874947367562688414176727"
            "1834370510168906693914817658458601172748263063804952000917508074099"
        ),
    },
    "RSA-704": {
        "n": (
            "481129598370820486061910647157532651356796844697246869285079385492"
            "987722656027353694173131020810025627287867430862203919494504712371"
            "619121195977550391096736944270521294349788304367403252976432190401"
            "969123"
        ),
        "p": (
            "19865132044017017703881887474419596953590950446010122769055743392513"
            "2546433239724739321584520666152954911097468590654663243216139024431"
        ),
        "q": (
            "24268746322910065808925925994438204525357948974397706303431353675179"
            "3336070696298229635385621259943455710335945863978218992159281201019"
        ),
    },
    "RSA-768": {
        "n": (
            "12301866845301177551304949583849627207728535695953347921973224521517264005072"
            "63657518745202199786469389956474942774063845925192557326303453731548268507917"
            "026122142913461670429214311602221240479274737794080665351419597459856902143413"
        ),
        "p": (
            "334780716989568987860441698482126908177047949837137685689124313889315331684923"
            "1738672651160999995375399"
        ),
        "q": (
            "367460436667995904282446337996279526322791581643430876426760228381573966651127"
            "9233373417143396810270092798736308917"
        ),
    },
}


@classifier(
    label="RSA challenge number",
    description="One of the published RSA Challenge semiprimes (e.g. RSA-100, RSA-110, RSA-120, RSA-129, …).",
    category=CATEGORY,
)
def is_rsa_challenge(n: int) -> tuple[bool, str | None]:
    """
    Recognises selected historical RSA Challenge semiprimes and reports which RSA-k
    instance it is, together with an abbreviated display of its prime factors.

    Uses string comparison against a small table so there is no performance impact
    on arbitrary n.
    """
    n_abs = abs(int(n))

    for name, rec in RSA_CHALLENGES.items():
        if n_abs == int(rec["n"]):
            # Convert to ints only for pretty-print; abbr_int_fast will keep it short.
            p = int(rec["p"])
            q = int(rec["q"])
            n_abbr = abbr_int_fast(n)
            p_abbr = (p)
            q_abbr = (q)
            details = (
                f"{name}: {n_abbr} = {p_abbr} × {q_abbr} "
            )
            return True, details

    return False, None


@classifier(
    label="Strong pseudoprime",
    description="Composite n passing the Miller–Rabin strong probable prime test for at least one base in {2,3,5,7,11,13}.",
    oeis="A001262",  # (base 2), A020299 (base3), A020231 (base 5) A020233 (base 7)
    category=CATEGORY,
    limit=10**1000 - 1
)
def _is_strong_pseudoprime(n: int, bases=BASES):
    """
    Strong (Miller–Rabin) pseudoprime test:
    Composite n passing the strong probable prime test for at least one base in bases.
    """
    if n < 3 or n % 2 == 0 or isprime(n):
        return False, None

    # Factor n-1 as d*2^s with d odd
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    passing_bases = []
    for a in bases:
        if gcd(a, n) != 1:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            passing_bases.append(a)
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                passing_bases.append(a)
                break

    if passing_bases:
        bases_str = ", ".join(map(str, passing_bases))
        return True, f"passes strong test bases {bases_str}"
    return False, None
