"""Middleware insertion helpers for @Next/@Prev anchor-based positioning.

This module is intentionally dependency-free (no tools, no config, no I/O)
so it can be imported by both ``factory.py`` and ``lead_agent/agent.py``
without triggering circular imports.
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware

from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware


def _insert_extra(chain: list[AgentMiddleware], extras: list[AgentMiddleware]) -> None:
    """Insert *extras* into *chain* using ``@Next``/``@Prev`` anchor declarations.

    Algorithm
    ---------
    1. Validate each extra: cannot have both ``_next_anchor`` and ``_prev_anchor``.
    2. Conflict detection: two extras targeting the same anchor class in the
       same direction raises ``ValueError``.  Same anchor, opposite directions
       also raise (use cross-extra anchoring instead).
    3. Unanchored extras are inserted just before ``ClarificationMiddleware``.
    4. Anchored extras are resolved iteratively to support cross-extra anchoring
       (where one extra's anchor is another extra that has not yet been inserted).
    5. If any anchor remains unresolvable after all iterations → ``ValueError``
       (reports circular dependency if detected).

    ``ClarificationMiddleware`` re-pinning
    --------------------------------------
    Callers are responsible for re-pinning ``ClarificationMiddleware`` to the
    absolute end of the chain if a ``@Next(ClarificationMiddleware)`` could have
    displaced it.  See ``_build_middlewares`` / ``_assemble_from_features``.
    """
    next_targets: dict[type, type] = {}
    prev_targets: dict[type, type] = {}

    anchored: list[tuple[AgentMiddleware, str, type]] = []
    unanchored: list[AgentMiddleware] = []

    for mw in extras:
        next_anchor = getattr(type(mw), "_next_anchor", None)
        prev_anchor = getattr(type(mw), "_prev_anchor", None)

        if next_anchor and prev_anchor:
            raise ValueError(
                f"{type(mw).__name__} cannot have both @Next and @Prev — choose one anchor direction"
            )

        if next_anchor:
            if next_anchor in next_targets:
                raise ValueError(
                    f"Conflict: {type(mw).__name__} and {next_targets[next_anchor].__name__} "
                    f"both declare @Next({next_anchor.__name__})"
                )
            if next_anchor in prev_targets:
                raise ValueError(
                    f"Conflict: {type(mw).__name__} @Next({next_anchor.__name__}) collides with "
                    f"{prev_targets[next_anchor].__name__} @Prev({next_anchor.__name__}) — "
                    f"use cross-anchoring between the two extras instead"
                )
            next_targets[next_anchor] = type(mw)
            anchored.append((mw, "next", next_anchor))

        elif prev_anchor:
            if prev_anchor in prev_targets:
                raise ValueError(
                    f"Conflict: {type(mw).__name__} and {prev_targets[prev_anchor].__name__} "
                    f"both declare @Prev({prev_anchor.__name__})"
                )
            if prev_anchor in next_targets:
                raise ValueError(
                    f"Conflict: {type(mw).__name__} @Prev({prev_anchor.__name__}) collides with "
                    f"{next_targets[prev_anchor].__name__} @Next({prev_anchor.__name__}) — "
                    f"use cross-anchoring between the two extras instead"
                )
            prev_targets[prev_anchor] = type(mw)
            anchored.append((mw, "prev", prev_anchor))

        else:
            unanchored.append(mw)

    # Unanchored → just before ClarificationMiddleware
    clar_idx = next(
        (i for i, m in enumerate(chain) if isinstance(m, ClarificationMiddleware)),
        len(chain),
    )
    for mw in unanchored:
        chain.insert(clar_idx, mw)
        clar_idx += 1

    # Anchored → iterative resolution (handles cross-extra anchoring)
    pending = list(anchored)
    max_rounds = len(pending) + 1
    for _ in range(max_rounds):
        if not pending:
            break
        still_pending = []
        for mw, direction, anchor in pending:
            idx = next(
                (i for i, m in enumerate(chain) if isinstance(m, anchor)),
                None,
            )
            if idx is None:
                still_pending.append((mw, direction, anchor))
                continue
            if direction == "next":
                chain.insert(idx + 1, mw)
            else:
                chain.insert(idx, mw)
        if len(still_pending) == len(pending):
            # No progress → either circular or genuinely missing
            anchor_classes = {a for _, _, a in still_pending}
            extra_classes = {type(m) for m, _, _ in still_pending}
            circular = anchor_classes & extra_classes
            names = [type(m).__name__ for m, _, _ in still_pending]
            if circular:
                raise ValueError(
                    f"Circular dependency among extra middlewares: "
                    f"{', '.join(t.__name__ for t in circular)}"
                )
            raise ValueError(
                f"Cannot resolve positions for {', '.join(names)} — "
                f"anchor class(es) {', '.join(a.__name__ for _, _, a in still_pending)} "
                f"not found in middleware chain"
            )
        pending = still_pending
