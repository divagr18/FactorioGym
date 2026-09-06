"""Python and Lua must share one protocol vocabulary (PLAN.md 1.1).

The two sides declare the wire vocabulary independently: Python as enums in
``protocol.py``, Lua as string literals in the mod. Nothing but this test
stops them drifting, and a drift shows up as an uncaught ``ValueError`` deep
inside ``Response.from_dict`` rather than as a structured protocol error.

Engine-free by design: it reads the Lua source, so it runs everywhere and on
every change, not only when a Factorio binary is available.
"""

from __future__ import annotations

import re

import pytest

from factoriorl.paths import mod_source_dir
from factoriorl.protocol import ActionStatus, ErrorCode, RequestType, ResultCode

#: Codes Python understands that the mod does not emit yet, each with the
#: phase that will introduce it. Anything not listed here must be live on both
#: sides -- the point is that an unused code is a deliberate reservation, not
#: something that quietly rotted.
RESERVED_CODES = {
    ErrorCode.COLLISION: "Phase 2.1: placement blocked by collision",
    ErrorCode.DUPLICATE_REQUEST: "reserved; duplicates use ResultCode.DUPLICATE today",
}

#: Worker-level status strings that are deliberately not ActionStatus values.
NON_ACTION_STATUS = {"ready"}


def _lua_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(mod_source_dir().rglob("*.lua"))
    )


@pytest.fixture(scope="module")
def lua() -> str:
    return _lua_sources()


def test_lua_error_codes_are_all_known_to_python(lua):
    emitted = set(re.findall(r'err\("([a-z_]+)"', lua)) | set(
        re.findall(r'code = "([a-z_]+)", message', lua)
    )
    known = {code.value for code in ErrorCode}
    assert emitted, "no error codes found in the Lua sources; the regex is wrong"
    assert emitted <= known, f"Lua emits error codes Python cannot parse: {sorted(emitted - known)}"


def test_python_error_codes_are_emitted_or_explicitly_reserved(lua):
    emitted = set(re.findall(r'err\("([a-z_]+)"', lua)) | set(
        re.findall(r'code = "([a-z_]+)"', lua)
    )
    reserved = {code.value for code in RESERVED_CODES}
    unused = {code.value for code in ErrorCode} - emitted - reserved
    assert not unused, f"ErrorCode values neither emitted by Lua nor reserved: {sorted(unused)}"


def test_lua_result_codes_are_all_known_to_python(lua):
    emitted = set(re.findall(r'respond\(request, "([a-z_]+)"', lua)) | set(
        re.findall(r'code = "([a-z_]+)",\n', lua)
    )
    known = {code.value for code in ResultCode}
    assert emitted, "no result codes found in the Lua sources; the regex is wrong"
    assert emitted <= known, (
        f"Lua emits result codes Python cannot parse: {sorted(emitted - known)}"
    )


def test_lua_action_statuses_are_all_known_to_python(lua):
    emitted = set(re.findall(r'status = "([a-z_]+)"', lua)) - NON_ACTION_STATUS
    known = {status.value for status in ActionStatus}
    assert emitted, "no status literals found in the Lua sources; the regex is wrong"
    assert emitted <= known, (
        f"Lua emits action statuses Python cannot parse: {sorted(emitted - known)}"
    )


def test_every_request_type_has_a_lua_handler(lua):
    handlers = re.search(r"local HANDLERS = \{(.*?)\n\}", lua, re.DOTALL)
    assert handlers, "HANDLERS table not found in the Lua runtime"
    registered = set(re.findall(r"^\s*([a-z_]+)\s*=", handlers.group(1), re.MULTILINE))
    expected = {request_type.value for request_type in RequestType}
    assert expected <= registered, (
        f"request types with no Lua handler: {sorted(expected - registered)}"
    )


def test_mutating_request_types_are_the_ones_python_expects(lua):
    mutating = re.search(r"local MUTATING = \{(.*?)\}", lua, re.DOTALL)
    assert mutating, "MUTATING table not found in the Lua runtime"
    declared = set(re.findall(r"([a-z_]+)\s*=\s*true", mutating.group(1)))
    # Everything that changes world or episode state must be deduplicated and
    # episode-checked; read-only types must not be, or repeated observation
    # would start returning stale stored responses.
    assert declared == {
        RequestType.ADVANCE.value,
        RequestType.ACT.value,
        RequestType.RESET.value,
    }
