from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumCtx:
    # --- non-default fields (no "= ...") FIRST ---
    n: int
    fac: dict[int, int]              # May include composite cofactors
    sigma: int | None                # None if incomplete or n in {0}
    tau: int | None                  # None if incomplete or n in {0}
    unitary_sigma: int | None        # None if incomplete or n in {0}
    unitary_tau: int | None          # None if incomplete or n in {0}
    omega: int | None                # None if incomplete or n in {0}

    # --- fields WITH defaults AFTER all non-defaults ---
    incomplete: bool = False         # True if fac contains any composite base
    composite_bases: tuple[int, ...] = ()  # optional: for UI/debug
    is_prime: bool | None = None     # lazy cache for primality of n

    @property
    def is_fully_factored(self) -> bool:
        return not self.incomplete
