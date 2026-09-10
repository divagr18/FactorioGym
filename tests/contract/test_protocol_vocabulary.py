"""Python and Lua must share one protocol vocabulary (PLAN.md 1.1, 2.1).

The two sides declare the wire contract independently: Python as enums in
``protocol.py`` and a mirror in ``action_matrix.py``, Lua as tables in
``protocol.lua`` and ``matrix.lua``. Nothing but this test stops them drifting,
and a drift surfaces as an uncaught ``ValueError`` deep inside
``Response.from_dict`` rather than as a structured protocol error.

Engine-free by design: it reads the Lua source, so it runs on every change
rather than only when a Factorio binary is available.
"""

from __future__ import annotations

import re

import pytest

from factoriorl.action_matrix import ACTIONS, ORDER
from factoriorl.paths import mod_source_dir
from factoriorl.protocol import (
    PROTOCOL_VERSION,
    ActionStatus,
    ActionType,
    ErrorCode,
    RequestType,
    ResultCode,
)

#: Codes Python understands that the mod does not emit, each a deliberate
#: reservation rather than something that quietly rotted.
RESERVED_CODES = {
    ErrorCode.DUPLICATE_REQUEST: "duplicates are reported via ResultCode.DUPLICATE",
}


def _read(name: str) -> str:
    return (mod_source_dir() / "factoriorl" / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def protocol_lua() -> str:
    return _read("protocol.lua")


@pytest.fixture(scope="module")
def matrix_lua() -> str:
    return _read("matrix.lua")


@pytest.fixture(scope="module")
def runtime_lua() -> str:
    return _read("runtime.lua")


def _table_values(source: str, table_name: str) -> set[str]:
    """Collect the string values of a `protocol.X = { KEY = "value" }` table."""
    match = re.search(rf"protocol\.{table_name} = \{{(.*?)\n\}}", source, re.DOTALL)
    assert match, f"table {table_name} not found in protocol.lua"
    return set(re.findall(r'=\s*"([a-z_]+)"', match.group(1)))


def test_protocol_version_matches_on_both_sides(protocol_lua):
    """One declaration each. These used to be able to drift silently."""
    match = re.search(r"protocol\.VERSION = (\d+)", protocol_lua)
    assert match, "protocol.VERSION not found"
    assert int(match.group(1)) == PROTOCOL_VERSION


def test_lua_error_codes_are_all_known_to_python(protocol_lua):
    declared = _table_values(protocol_lua, "ERR")
    known = {code.value for code in ErrorCode}
    assert declared, "no error codes found; the regex is wrong"
    assert declared <= known, f"Lua declares codes Python cannot parse: {sorted(declared - known)}"


def test_python_error_codes_are_declared_or_explicitly_reserved(protocol_lua):
    declared = _table_values(protocol_lua, "ERR")
    reserved = {code.value for code in RESERVED_CODES}
    unused = {code.value for code in ErrorCode} - declared - reserved
    assert not unused, f"ErrorCode values neither declared in Lua nor reserved: {sorted(unused)}"


def test_result_codes_agree(protocol_lua):
    assert _table_values(protocol_lua, "CODE") == {code.value for code in ResultCode}


def test_action_statuses_agree(protocol_lua):
    assert _table_values(protocol_lua, "STATUS") == {status.value for status in ActionStatus}


def test_every_request_type_has_a_lua_handler(runtime_lua):
    handlers = re.search(r"local HANDLERS = \{(.*?)\n\}", runtime_lua, re.DOTALL)
    assert handlers, "HANDLERS table not found"
    registered = set(re.findall(r"^\s*([a-z_]+)\s*=", handlers.group(1), re.MULTILINE))
    expected = {request_type.value for request_type in RequestType}
    assert expected <= registered, f"request types with no handler: {sorted(expected - registered)}"


def test_mutating_request_types_are_the_ones_python_expects(runtime_lua):
    mutating = re.search(r"local MUTATING = \{(.*?)\}", runtime_lua, re.DOTALL)
    assert mutating, "MUTATING table not found"
    declared = set(re.findall(r"([a-z_]+)\s*=\s*true", mutating.group(1)))
    # Everything that changes world or episode state must be deduplicated and
    # episode-checked; read-only types must not be, or repeated observation
    # would start returning stale stored responses.
    assert declared == {
        RequestType.ADVANCE.value,
        RequestType.ACT.value,
        RequestType.RESET.value,
        RequestType.STEP.value,
        # `disrupt` writes to an installed scene, so it needs both properties
        # this table confers. Deduplication: a retried disruption must return
        # the stored reply rather than empty a second fuel inventory. Episode
        # check: a disruption declared for the previous episode must be refused
        # rather than applied to this one.
        RequestType.DISRUPT.value,
        # `open_world` destroys player-force entities, moves the character and
        # inserts a starting inventory, so it needs both properties for the same
        # reasons `reset` does. Deduplication in particular: a retried
        # initialisation must return the stored reply rather than insert a second
        # copy of freeplay's items.
        RequestType.OPEN_WORLD.value,
    }


def test_action_names_agree(matrix_lua):
    order = re.search(r"matrix\.ORDER = \{(.*?)\n\}", matrix_lua, re.DOTALL)
    assert order, "matrix.ORDER not found"
    lua_names = re.findall(r'"([a-z_]+)"', order.group(1))
    assert lua_names == [action.value for action in ORDER], (
        "matrix.lua and action_matrix.py disagree on the action set or its order"
    )
    assert set(lua_names) == {action.value for action in ActionType}


def test_every_action_has_a_lua_handler(matrix_lua):
    actions_lua = _read("actions.lua")
    for action in ActionType:
        assert re.search(rf"^H\.{action.value} = function", actions_lua, re.MULTILINE), (
            f"no handler for action {action.value}"
        )


def test_matrix_declares_the_same_ongoing_and_reach(matrix_lua):
    for action, spec in ACTIONS.items():
        block = re.search(rf"\n  {action.value} = \{{(.*?)\n  \}},", matrix_lua, re.DOTALL)
        assert block, f"matrix.lua has no entry for {action.value}"
        body = block.group(1)
        ongoing = re.search(r"ongoing = (true|false)", body)
        assert ongoing and (ongoing.group(1) == "true") == spec.ongoing, (
            f"{action.value}: ongoing disagrees between Lua and Python"
        )
        cancellable = re.search(r"cancellable = (true|false)", body)
        assert cancellable and (cancellable.group(1) == "true") == spec.cancellable, (
            f"{action.value}: cancellable disagrees"
        )
        reach = re.search(r"reach = matrix\.REACH\.([A-Z]+)", body)
        assert reach and reach.group(1).lower() == spec.reach.value, (
            f"{action.value}: reach disagrees"
        )


def test_matrix_required_fields_agree(matrix_lua):
    for action, spec in ACTIONS.items():
        block = re.search(rf"\n  {action.value} = \{{(.*?)\n  \}},", matrix_lua, re.DOTALL)
        body = block.group(1)
        payload = re.search(r"payload = \{(.*?)\n    \}", body, re.DOTALL)
        if not spec.required and not spec.optional:
            continue
        assert payload, f"{action.value}: payload block not found"
        required = set(re.findall(r"(\w+) = \{[^}]*required = true", payload.group(1)))
        assert required == set(spec.required), (
            f"{action.value}: required fields disagree "
            f"(lua={sorted(required)}, python={sorted(spec.required)})"
        )


def test_failure_codes_are_real_error_codes():
    for action, spec in ACTIONS.items():
        for code in spec.failure_codes:
            assert isinstance(code, ErrorCode), f"{action.value} declares a non-ErrorCode"
