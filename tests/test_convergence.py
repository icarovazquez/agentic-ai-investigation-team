import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")
from investigation_team.agents.evidence_analyst import EvidenceAnalysis, HypothesisAssessment
from investigation_team.convergence import check_convergence, residual_concerns
from investigation_team.domain import HypothesisStatus as S


def A(**conf):  # conf: h1=("supported", 0.9) ...
    return EvidenceAnalysis(incident_id="i", assessments=[
        HypothesisAssessment(hypothesis_id=h, status=S(st), confidence=c, rationale="r") for h, (st, c) in conf.items()])


GOOD = dict(h1=("supported", 0.92), h2=("weakened", 0.25), h3=("rejected", 0.05))


def test_two_stable_rounds_converge():
    assert check_convergence([A(**GOOD), A(**GOOD)], 0.8) == "h1"


def test_one_round_is_not_enough_and_zero_disables():
    assert check_convergence([A(**GOOD)], 0.8) is None
    assert check_convergence([A(**GOOD), A(**GOOD)], 0.8, stable_rounds=0) is None


def test_leader_change_or_below_bar_blocks():
    flip = dict(h1=("weakened", 0.3), h2=("supported", 0.9), h3=("rejected", 0.05))
    assert check_convergence([A(**flip), A(**GOOD)], 0.8) is None
    low = dict(h1=("supported", 0.7), h2=("weakened", 0.2))
    assert check_convergence([A(**GOOD), A(**low)], 0.8) is None


def test_credible_rival_blocks():
    rival = dict(h1=("supported", 0.9), h2=("proposed", 0.6))
    assert check_convergence([A(**rival), A(**rival)], 0.8) is None
    # a rejected hypothesis with an odd confidence is not a rival; directional_confidence floors it
    odd = dict(h1=("supported", 0.9), h2=("rejected", 0.9))
    assert check_convergence([A(**odd), A(**odd)], 0.8) == "h1"


def test_only_last_n_rounds_count():
    bad = dict(h1=("proposed", 0.5), h2=("proposed", 0.5))
    assert check_convergence([A(**bad), A(**GOOD), A(**GOOD)], 0.8) == "h1"


def test_residual_concerns_kept():
    from investigation_team.agents.hypothesis_challenger import ChallengeReport, HypothesisChallenge
    rep = ChallengeReport(incident_id="i", challenges=[HypothesisChallenge(
        hypothesis_id="h1", original_status="supported", original_confidence=0.9, challenge_outcome="disputed",
        rebuttal="no optical readings", alternative_explanation=None, additional_evidence_needed=["optical power"])],
        systemic_concerns=["only one path tested"])
    out = residual_concerns(rep, "h1")
    assert any("no optical readings" in c for c in out) and any("optical power" in c for c in out) and any("only one path" in c for c in out)
    assert residual_concerns(None, "h1") == []
