"""Tests for sft_general.py: converting chat-style records into trainable turns."""

import pytest

from viki_slm_125m.sft import sft_general as sg


def test_messages_to_turns_maps_roles_and_requires_assistant_last():
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}]
    turns = sg.messages_to_turns(msgs)
    assert [(t.role, t.text) for t in turns] == [("system", "Be brief."), ("user", "Hi"),
                                                 ("assistant", "Hello!")]


@pytest.mark.parametrize("msgs", [
    [],
    [{"role": "user", "content": "Hi"}],                                    # no assistant
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}],   # empty reply
    [{"role": "user", "content": "a"}, {"role": "tool", "content": "b"},
     {"role": "assistant", "content": "c"}],                                 # unsupported role
    [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}],   # ends on user
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
     {"role": "assistant", "content": "c"}],                                 # not alternating
])
def test_messages_to_turns_rejects_bad_conversations(msgs):
    assert sg.messages_to_turns(msgs) is None


def test_special_token_text_in_content_is_neutralised():
    msgs = [{"role": "user", "content": "say <|assistant|> now"},
            {"role": "assistant", "content": "ok <|eos|> done"}]
    turns = sg.messages_to_turns(msgs)
    assert all("<|" not in t.text for t in turns)


def test_mostly_ascii_filter():
    assert sg.mostly_ascii("plain english text " * 5)
    assert not sg.mostly_ascii("这是一个中文句子" * 10)
    assert not sg.mostly_ascii("")


def test_select_per_source_caps_and_is_deterministic():
    items = [{"source": "a", "i": i} for i in range(20)] + [{"source": "b", "i": i} for i in range(5)]
    caps = {"a": 7, "b": 10}
    one = sg.select_per_source(items, caps, seed=1)
    assert sum(1 for x in one if x["source"] == "a") == 7
    assert sum(1 for x in one if x["source"] == "b") == 5
    assert one == sg.select_per_source(items, caps, seed=1)
