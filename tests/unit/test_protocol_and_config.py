"""Unit tests: protocol types, RCON framing, config generation. No engine."""

from __future__ import annotations

import json

import pytest

from factoriorl.protocol import (
    PROTOCOL_VERSION,
    ActionStatus,
    ErrorCode,
    Request,
    RequestType,
    Response,
    ResultCode,
)
from factoriorl.rcon import encode_packet, lua_string, wrap_lua
from factoriorl.worker_config import MAP_SETTINGS, WorkerPorts, WorkerSpec


def test_protocol_version_is_one():
    assert PROTOCOL_VERSION == 1


def test_request_roundtrip_carries_identifiers():
    req = Request("r-1", "ep-1", RequestType.ACT, {"action": "wait"})
    raw = json.loads(req.to_json())
    assert raw["protocol"] == 1
    assert raw["request_id"] == "r-1"
    assert raw["episode_id"] == "ep-1"
    assert raw["type"] == "act"
    assert raw["payload"]["action"] == "wait"


def test_response_roundtrip_with_error():
    resp = Response.from_dict(
        {
            "protocol": 1,
            "request_id": "r-2",
            "episode_id": "ep-1",
            "code": "rejected",
            "error": {"code": "out_of_reach", "message": "too far"},
        }
    )
    assert resp.code is ResultCode.REJECTED
    assert resp.error.code is ErrorCode.OUT_OF_REACH
    assert not resp.ok


def test_response_ok_result():
    resp = Response.from_dict(
        {
            "protocol": 1,
            "request_id": "r",
            "episode_id": "e",
            "code": "ok",
            "result": {"status": "completed"},
        }
    )
    assert resp.ok
    assert resp.result["status"] == ActionStatus.COMPLETED.value


def test_lua_string_escapes_are_lua_safe():
    text = 'say "hi"\n\t\\done'
    quoted = lua_string(text)
    assert quoted == '"say \\"hi\\"\\n\\t\\\\done"'
    # No \u escapes: Lua 5.2 cannot parse them.
    assert "\\u" not in lua_string("\u00e9\u4e2d")


def test_wrap_lua_embeds_code():
    wrapped = wrap_lua("return 1")
    assert 'remote.call("frrl_bridge", "run"' in wrapped
    assert '"return 1"' in wrapped


def test_encode_packet_double_nul():
    packet = encode_packet(10, 2, "/c return 1")
    # length field covers id+type+body+2 NULs.
    import struct

    length, pid, ptype = struct.unpack_from("<iii", packet, 0)
    assert (pid, ptype) == (10, 2)
    assert packet.endswith(b"\x00\x00")
    assert length == 8 + len(b"/c return 1") + 2


def test_map_settings_disable_hostile_systems():
    assert MAP_SETTINGS["pollution"]["enabled"] is False
    assert MAP_SETTINGS["enemy_evolution"]["enabled"] is False
    assert MAP_SETTINGS["enemy_expansion"]["enabled"] is False


def test_worker_spec_generates_isolated_configs(tmp_path, monkeypatch):
    import factoriorl.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    spec = WorkerSpec(worker_id="iso", ports=WorkerPorts(game=1000, rcon=1001), map_seed=7)
    spec.write_config_files()
    config_ini = spec.config_ini.read_text(encoding="utf-8")
    assert "write-data=" in config_ini and "iso" in config_ini
    server = json.loads(spec.server_settings.read_text(encoding="utf-8"))
    assert server["auto_pause"] is False
    assert server["allow_commands"] == "true"
    gen = json.loads(spec.map_gen_settings.read_text(encoding="utf-8"))
    assert gen["peaceful_mode"] is True
    assert gen["seed"] == 7
    assert gen["autoplace_controls"]["enemy-base"]["frequency"] == 0


def test_worker_id_validation():
    from factoriorl.paths import worker_dir

    with pytest.raises(ValueError):
        worker_dir("../escape")
    with pytest.raises(ValueError):
        worker_dir("a/b")
