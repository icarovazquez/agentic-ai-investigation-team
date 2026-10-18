import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")
from types import SimpleNamespace as NS
from investigation_team import llm


def _resp(text, reason):
    return NS(choices=[NS(message=NS(content=text), finish_reason=reason)], usage=NS(prompt_tokens=1, completion_tokens=1))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = NS(completions=NS(create=self._create))
    def _create(self, **kw):
        self.calls.append(kw["max_tokens"])
        return self.replies.pop(0)


def test_truncated_output_retried_once_with_double_budget(monkeypatch):
    fake = FakeClient([_resp('{"a": [1,', "length"), _resp('{"a": [1,2]}', "stop")])
    monkeypatch.setattr(llm, "Client", fake)
    out = llm.llm_call("evidence_analyst_agent", [{"role": "user", "content": "x"}], max_tokens=2000)
    assert fake.calls == [2000, 4000] and out["content"] == '{"a": [1,2]}' and out["truncated"] is False


def test_normal_output_not_retried(monkeypatch):
    fake = FakeClient([_resp("ok", "stop")])
    monkeypatch.setattr(llm, "Client", fake)
    llm.llm_call("evidence_analyst_agent", [{"role": "user", "content": "x"}], max_tokens=2000)
    assert fake.calls == [2000]


def test_retry_is_bounded(monkeypatch):
    fake = FakeClient([_resp("cut", "length"), _resp("still cut", "length")])
    monkeypatch.setattr(llm, "Client", fake)
    out = llm.llm_call("evidence_analyst_agent", [{"role": "user", "content": "x"}], max_tokens=2000)
    assert len(fake.calls) == 2 and out["truncated"] is True
