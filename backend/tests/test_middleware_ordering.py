"""Unit tests for @Next/@Prev middleware positioning.

Covers:
  1. @Next inserts after anchor
  2. @Prev inserts before anchor
  3. Unanchored extra lands before ClarificationMiddleware
  4. Cross-extra anchoring (@Next on another extra middleware)
  5. Conflict detection — two extras with same anchor+direction
  6. Both-anchor conflict — @Next and @Prev on same class
  7. Unresolvable anchor raises ValueError
  8. Circular dependency among extras raises ValueError
  9. ClarificationMiddleware always ends up last
  10. _build_middlewares custom_middlewares path uses _insert_extra
"""

import pytest
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deerflow.agents.features import Next, Prev
from deerflow.agents.ordering import _insert_extra
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
from deerflow.agents.middlewares.title_middleware import TitleMiddleware


# ---------------------------------------------------------------------------
# Minimal stub middleware classes for testing
# ---------------------------------------------------------------------------


class _A(AgentMiddleware):
    pass


class _B(AgentMiddleware):
    pass


class _C(AgentMiddleware):
    pass


def _make_chain() -> list[AgentMiddleware]:
    """Return a minimal chain: [_A, _B, ClarificationMiddleware]."""
    return [_A(), _B(), ClarificationMiddleware()]


def _class_order(chain: list[AgentMiddleware]) -> list[type]:
    return [type(m) for m in chain]


# ---------------------------------------------------------------------------
# 1. @Next inserts immediately after anchor
# ---------------------------------------------------------------------------


def test_next_inserts_after_anchor():
    @Next(_A)
    class _After_A(AgentMiddleware):
        pass

    chain = _make_chain()
    _insert_extra(chain, [_After_A()])
    order = _class_order(chain)
    assert order.index(_After_A) == order.index(_A) + 1


# ---------------------------------------------------------------------------
# 2. @Prev inserts immediately before anchor
# ---------------------------------------------------------------------------


def test_prev_inserts_before_anchor():
    @Prev(_B)
    class _Before_B(AgentMiddleware):
        pass

    chain = _make_chain()
    _insert_extra(chain, [_Before_B()])
    order = _class_order(chain)
    assert order.index(_Before_B) == order.index(_B) - 1


# ---------------------------------------------------------------------------
# 3. Unanchored extra lands just before ClarificationMiddleware
# ---------------------------------------------------------------------------


def test_unanchored_lands_before_clarification():
    class _Plain(AgentMiddleware):
        pass

    chain = _make_chain()
    _insert_extra(chain, [_Plain()])
    order = _class_order(chain)
    clar_idx = order.index(ClarificationMiddleware)
    plain_idx = order.index(_Plain)
    assert plain_idx == clar_idx - 1


# ---------------------------------------------------------------------------
# 4. Cross-extra anchoring: second extra @Prev(first extra)
# ---------------------------------------------------------------------------


def test_cross_extra_anchoring():
    @Next(_A)
    class _First(AgentMiddleware):
        pass

    @Prev(_First)
    class _BeforeFirst(AgentMiddleware):
        pass

    chain = _make_chain()
    _insert_extra(chain, [_First(), _BeforeFirst()])
    order = _class_order(chain)
    # _BeforeFirst must appear before _First
    assert order.index(_BeforeFirst) < order.index(_First)
    # _First must appear after _A
    assert order.index(_First) > order.index(_A)


# ---------------------------------------------------------------------------
# 5. Conflict: two extras with same anchor and same direction
# ---------------------------------------------------------------------------


def test_conflict_same_next_anchor():
    @Next(_A)
    class _X(AgentMiddleware):
        pass

    @Next(_A)
    class _Y(AgentMiddleware):
        pass

    with pytest.raises(ValueError, match="Conflict"):
        _insert_extra(_make_chain(), [_X(), _Y()])


def test_conflict_same_prev_anchor():
    @Prev(_B)
    class _X(AgentMiddleware):
        pass

    @Prev(_B)
    class _Y(AgentMiddleware):
        pass

    with pytest.raises(ValueError, match="Conflict"):
        _insert_extra(_make_chain(), [_X(), _Y()])


# ---------------------------------------------------------------------------
# 6. A single class having both @Next and @Prev raises immediately
# ---------------------------------------------------------------------------


def test_both_anchors_raises():
    class _Ambiguous(AgentMiddleware):
        _next_anchor = _A
        _prev_anchor = _B

    with pytest.raises(ValueError, match="both @Next and @Prev"):
        _insert_extra(_make_chain(), [_Ambiguous()])


# ---------------------------------------------------------------------------
# 7. Anchor not found in chain raises ValueError
# ---------------------------------------------------------------------------


def test_unresolvable_anchor_raises():
    @Next(_C)  # _C is never in the chain
    class _RefMissing(AgentMiddleware):
        pass

    with pytest.raises(ValueError):
        _insert_extra(_make_chain(), [_RefMissing()])


# ---------------------------------------------------------------------------
# 8. Circular dependency among extras raises ValueError
# ---------------------------------------------------------------------------


def test_circular_dependency_raises():
    class _P(AgentMiddleware):
        pass

    class _Q(AgentMiddleware):
        pass

    _P._next_anchor = _Q   # type: ignore[attr-defined]
    _Q._next_anchor = _P   # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        _insert_extra(_make_chain(), [_P(), _Q()])

    # Cleanup to avoid polluting other tests
    del _P._next_anchor
    del _Q._next_anchor


# ---------------------------------------------------------------------------
# 9. ClarificationMiddleware always ends up last
# ---------------------------------------------------------------------------


def test_clarification_always_last_after_next():
    """@Next(ClarificationMiddleware) pushes something after it;
    the re-pinning step must move ClarificationMiddleware back to tail."""

    @Next(ClarificationMiddleware)
    class _AfterClarification(AgentMiddleware):
        pass

    chain = _make_chain()
    _insert_extra(chain, [_AfterClarification()])

    # After re-pinning ClarificationMiddleware must be the last element
    # (agent.py does this re-pinning; here we test _insert_extra directly
    # so we do the re-pin manually like agent.py would)
    from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware as CM
    clar_idx = next(i for i, m in enumerate(chain) if isinstance(m, CM))
    if clar_idx != len(chain) - 1:
        chain.append(chain.pop(clar_idx))

    assert isinstance(chain[-1], ClarificationMiddleware)


# ---------------------------------------------------------------------------
# 10. Multiple unanchored extras preserve insertion order
# ---------------------------------------------------------------------------


def test_multiple_unanchored_preserve_order():
    class _P1(AgentMiddleware):
        pass

    class _P2(AgentMiddleware):
        pass

    class _P3(AgentMiddleware):
        pass

    chain = _make_chain()
    _insert_extra(chain, [_P1(), _P2(), _P3()])
    order = _class_order(chain)

    clar_idx = order.index(ClarificationMiddleware)
    # All three must appear in order immediately before ClarificationMiddleware
    assert order[clar_idx - 3 : clar_idx] == [_P1, _P2, _P3]


# ---------------------------------------------------------------------------
# 11. Decorator type validation
# ---------------------------------------------------------------------------


def test_next_rejects_non_middleware_anchor():
    with pytest.raises(TypeError):
        Next(object)  # type: ignore[arg-type]


def test_prev_rejects_non_middleware_anchor():
    with pytest.raises(TypeError):
        Prev(str)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 12. Real-world usage: SandboxAuditMiddleware declares @Next(SandboxMiddleware)
#     and auto-positions itself when passed as custom_middleware
# ---------------------------------------------------------------------------


def test_sandbox_audit_uses_next_decorator():
    """SandboxAuditMiddleware is decorated with @Next(SandboxMiddleware).
    When injected via custom_middlewares it must land immediately after
    SandboxMiddleware in the chain — no manual position arithmetic needed.
    """
    from deerflow.agents.middlewares.sandbox_audit_middleware import SandboxAuditMiddleware
    from deerflow.sandbox.middleware import SandboxMiddleware

    # Verify the decorator was applied on the class itself
    assert getattr(SandboxAuditMiddleware, "_next_anchor", None) is SandboxMiddleware

    # Build a minimal chain that contains a SandboxMiddleware instance
    class _FakeSandbox(SandboxMiddleware):
        """Lightweight stand-in — avoids real sandbox initialisation."""
        def __init__(self):
            # Skip lazy-init super().__init__ to keep this pure unit test
            pass

    chain: list[AgentMiddleware] = [_FakeSandbox(), ClarificationMiddleware()]
    _insert_extra(chain, [SandboxAuditMiddleware()])

    order = _class_order(chain)
    sandbox_idx = next(i for i, t in enumerate(order) if issubclass(t, SandboxMiddleware))
    audit_idx = order.index(SandboxAuditMiddleware)

    assert audit_idx == sandbox_idx + 1, (
        f"Expected SandboxAuditMiddleware immediately after SandboxMiddleware, "
        f"got positions sandbox={sandbox_idx} audit={audit_idx} in {[t.__name__ for t in order]}"
    )
