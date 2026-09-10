"""The append-only history, and the one invariant that makes caching work.

A provider that caches on an exact prefix charges the cache-hit rate only for
the leading tokens that match a request it has already seen. So the property
worth testing is not "the messages look right" but literally: **request N's
whole body is a prefix of request N+1's body.** Everything else here exists to
protect that.

The stake, from `factoriorl.pricing`: 50x between a hit and a miss, which over a
300-decision run is $1.19 against $34.15.
"""

from __future__ import annotations

from factoriorl.agent.transcript import Transcript


def test_a_sent_body_is_a_prefix_of_the_next_one() -> None:
    """The whole design in one assertion."""
    transcript = Transcript(system="rules")
    transcript.append_user("observation 1")
    first = transcript.messages()

    transcript.append_assistant('{"action": 0}')
    transcript.append_user("observation 2")
    second = transcript.messages()

    assert second[: len(first)] == first
    assert len(second) == len(first) + 2


def test_the_static_prefix_rides_with_the_system_message() -> None:
    """One message, not two: a provider that normalises or merges a second
    system message would silently change the very bytes being cached."""
    transcript = Transcript(system="rules", static_prefix="the catalog")
    body = transcript.messages()
    assert len(body) == 1
    assert body[0]["role"] == "system"
    assert body[0]["content"] == "rules\n\nthe catalog"


def test_a_rejection_appends_the_answer_and_the_reason() -> None:
    """A retry must extend the history, never rewrite the turn already sent.

    Splicing a correction into the previous user message -- which is what the
    loop used to do -- invalidates the cached prefix from that point on.
    """
    transcript = Transcript(system="rules")
    transcript.append_user("observation")
    before = transcript.messages()

    transcript.append_rejection("not json", "unparseable: no object found")
    after = transcript.messages()

    assert after[: len(before)] == before, "the sent turn was mutated"
    assert [m["role"] for m in after[-2:]] == ["assistant", "user"]
    assert after[-2]["content"] == "not json"


def test_the_user_turn_is_never_edited_after_it_is_sent() -> None:
    transcript = Transcript(system="rules")
    transcript.append_user("observation")
    original = transcript.messages()[-1]["content"]
    for reason in ("first", "second"):
        transcript.append_rejection("bad", reason)
    assert transcript.messages()[1]["content"] == original


def test_prefix_of_detects_a_mutation() -> None:
    """Guard the guard: the check has to be able to fail."""
    transcript = Transcript(system="rules")
    transcript.append_user("observation 1")
    body = transcript.messages()
    assert transcript.prefix_of(body + [{"role": "assistant", "content": "ok"}])

    tampered = [dict(m) for m in body]
    tampered[-1]["content"] = "observation 1 (edited)"
    assert not transcript.prefix_of(tampered + [{"role": "assistant", "content": "ok"}])


def test_only_content_is_kept_from_an_assistant_turn() -> None:
    """DeepSeek's thinking guide: without `tools`, `reasoning_content` "does not
    need to be passed back; even if passed to the API, it will be ignored and
    will not be concatenated into the context". Our contract sends no `tools`,
    so the transcript has no place to put it -- by construction, not by care."""
    transcript = Transcript(system="rules")
    transcript.append_assistant("the answer")
    message = transcript.messages()[-1]
    assert set(message) == {"role", "content"}


def test_a_reset_is_recorded_because_it_throws_the_cache_away() -> None:
    transcript = Transcript(system="rules", static_prefix="catalog")
    transcript.append_user("one")
    transcript.append_assistant("two")
    transcript.reset("compaction")

    assert transcript.turns == []
    assert transcript.resets == [{"reason": "compaction", "turns_dropped": 2}]
    # The static prefix survives by default: it is the part worth keeping cached.
    assert transcript.static_prefix == "catalog"
    assert transcript.to_dict()["cache_resets"][0]["reason"] == "compaction"


def test_recording_a_send_reports_the_body_it_sent() -> None:
    transcript = Transcript(system="rules")
    transcript.append_user("one")
    body = transcript.record_sent()
    assert body == transcript.messages()
    assert transcript.to_dict()["requests_sent"] == 1


def test_the_discipline_is_stated_in_the_artifact() -> None:
    """A reader of a run should be able to see what was promised about caching
    without reading this module."""
    described = Transcript(system="rules").to_dict()["discipline"]
    assert "append-only" in described
    assert "reasoning_content" in described
