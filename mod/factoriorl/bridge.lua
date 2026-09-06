-- RCON bridge: the only surface Python talks to.
--
-- dispatch(json_string): typed protocol entry point; closed dispatch table, no
--   arbitrary execution reachable from typed requests.
-- run(lua_code): evaluator-only arbitrary Lua (scenario setup, fault injection).
--   Never exposed through the typed protocol.
--
-- Both return plain Lua values; the Python side wraps them in table_to_json.

local bridge = {}

function bridge.register(runtime)
  remote.add_interface("frrl_bridge", {
    dispatch = function(json_string)
      return runtime.handle_json(json_string)
    end,
    run = function(lua_code)
      local fn, compile_err = loadstring(lua_code)
      if not fn then
        return { error = "compile: " .. tostring(compile_err) }
      end
      local ok, result = pcall(fn)
      if not ok then
        return { error = tostring(result) }
      end
      return { result = result }
    end,
  })
end

return bridge
