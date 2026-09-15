"""R2.2: a factorized policy over the shared contract.

The gate: "RL and LLM adapters can issue equivalent valid semantic actions and
produce equivalent state changes from matched states. Tests establish
argument/mask consistency and absence of privileged-information shortcuts. A
bounded training smoke check verifies sampling, log probabilities, and PPO
updates for the new distribution."
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
from stable_baselines3.common.vec_env import VecEnv

from factoriorl import catalog as catalog_module
from factoriorl import encoders
from factoriorl.env import FactorioEnv
from factoriorl.parameterized import (
    DIMENSIONS,
    UNUSED,
    ParameterizedEnv,
)
from factoriorl.tasks import get


class _Inner(FactorioEnv):
    """A `FactorioEnv` reduced to the parts the adapter reads."""

    def __init__(self, observation=None, catalog="parameterized-v1"):
        self.catalog = catalog_module.resolve(catalog)
        self.spec_ = get("repair_belt").spec
        self._observation = observation if observation is not None else _observation()
        self._steps = 0
        self._truth = {}
        self.observation_space = encoders.observation_space()
        self.issued: list[tuple[int, dict]] = []

    def step_arguments(self, action, arguments):
        self.issued.append((int(action), dict(arguments)))
        return self._encoded(), 0.0, False, False, {"action_key": "x"}

    def step(self, action):
        self.issued.append((int(action), {}))
        return self._encoded(), 0.0, False, False, {"action_key": "wait"}

    def _encoded(self):
        return encoders.encode(self._observation, np.zeros(encoders.GOAL_FEATURES))


def _observation(**overrides):
    base = {
        "character": {"position": [0.0, 0.0]},
        "entities": [
            {"h": "e1", "p": [3.5, 0.5], "type": "transport-belt"},
            {"h": "e2", "p": [-3.5, 0.5], "type": "container"},
        ],
        "resources": {"tiles": [{"h": "r1", "name": "iron-ore", "p": [2.5, 3.5]}]},
        "terrain": {"blocked": []},
        "inventory": {"transport-belt": 5},
        "inflight": [],
    }
    base.update(overrides)
    return base


def _env(**overrides):
    return ParameterizedEnv(_Inner(_observation(**overrides)))


def _vector(env, **choices):
    names = [name for name, _ in DIMENSIONS]
    return [choices.get(name, UNUSED) for name in names]


class TestSpaceAndMaskShape:
    def test_the_mask_is_a_flat_concatenation_of_nvec(self):
        """The shape sb3 splits columnwise; a mismatch fails silently."""
        env = _env()
        assert env.action_masks().shape[0] == int(env.action_space.nvec.sum())

    def test_no_dimension_is_ever_entirely_masked(self):
        """An all-false sub-mask becomes a uniform draw over illegal values."""
        for observation in (
            {},
            _observation(entities=[], resources={"tiles": []}, inventory={}),
            _observation(inventory={}),
        ):
            env = ParameterizedEnv(_Inner(observation or _observation()))
            mask = env.action_masks()
            offset = 0
            for size in env.action_space.nvec:
                assert mask[offset : offset + size].any(), "a dimension had no legal value"
                offset += int(size)

    def test_the_unused_sentinel_is_always_legal(self):
        env = _env()
        mask = env.action_masks()
        offset = int(env.action_space.nvec[0])
        for size in env.action_space.nvec[1:]:
            assert mask[offset + UNUSED]
            offset += int(size)

    def test_only_held_items_are_legal(self):
        env = _env()
        mask = env.action_masks()
        offset = int(sum(env.action_space.nvec[:4]))
        item_mask = mask[offset : offset + int(env.action_space.nvec[4])]
        legal = {encoders.ITEMS[i - 1] for i in np.flatnonzero(item_mask) if i != UNUSED}
        assert legal == {"transport-belt"}


class TestDecode:
    def test_a_full_vector_becomes_a_semantic_action(self):
        env = _env()
        op = env.env.catalog.keys().index("place_at")
        values = env._domain_values()
        item = encoders.ITEMS.index("transport-belt") + 1
        vector = _vector(env, operation=op, placement=1, direction=2, item=item)
        operation, arguments, failure = env.decode(vector)
        assert failure is None
        assert operation == op
        assert arguments["item"] == "transport-belt"
        assert arguments["position"] == values["placement"][0]
        assert arguments["direction"] == values["direction"][1]

    def test_a_missing_argument_is_named_not_silently_substituted(self):
        env = _env()
        op = env.env.catalog.keys().index("place_at")
        _, _, failure = env.decode(_vector(env, operation=op))
        assert failure and "left unused" in failure

    def test_an_argument_with_no_dimension_is_named(self):
        """`recipe` is not observable, so no dimension can offer a value."""
        env = _env()
        op = env.env.catalog.keys().index("craft_recipe")
        _, _, failure = env.decode(_vector(env, operation=op, amount=1))
        assert failure and "recipe" in failure

    def test_a_decode_failure_becomes_a_counted_no_op(self):
        env = _env()
        op = env.env.catalog.keys().index("place_at")
        _, _, _, _, info = env.step(_vector(env, operation=op))
        assert info["decode_failure"]
        assert env.decode_failures == 1
        assert env.env.issued[-1][0] == env.env.catalog.wait_index

    def test_an_operation_needing_no_argument_decodes_cleanly(self):
        env = _env()
        op = env.env.catalog.keys().index("wait")
        operation, arguments, failure = env.decode(_vector(env, operation=op))
        assert failure is None and arguments == {} and operation == op


class TestEquivalenceWithTheAddressedPath:
    """The gate: both clients issue equivalent semantic actions."""

    def test_the_rl_adapter_and_a_named_target_produce_the_same_payload(self):
        env = _env()
        template = next(t for t in env.env.catalog.templates if t.key == "rotate_at")
        op = env.env.catalog.keys().index("rotate_at")

        # RL: pick target index 1, which is handle e1.
        _, arguments, failure = env.decode(_vector(env, operation=op, target=1))
        assert failure is None
        rl_payload = template.bind({}, arguments)

        # LLM: name the handle directly, the way AgentLoop._execute does.
        llm_payload = template.bind({}, {"handle": "e1"})
        assert rl_payload == llm_payload

    def test_placement_agrees_between_the_two_paths(self):
        env = _env()
        template = next(t for t in env.env.catalog.templates if t.key == "place_at")
        op = env.env.catalog.keys().index("place_at")
        values = env._domain_values()
        item = encoders.ITEMS.index("transport-belt") + 1
        _, arguments, _ = env.decode(
            _vector(env, operation=op, placement=3, direction=1, item=item)
        )
        rl = template.bind({}, arguments)
        llm = template.bind(
            {},
            {
                "item": "transport-belt",
                "position": values["placement"][2],
                "direction": values["direction"][0],
            },
        )
        assert rl == llm


class TestNoPrivilegedInformation:
    def test_every_domain_comes_from_the_observation(self):
        """Two envs with the same observation must offer the same domains."""
        first = _env()
        second = _env()
        assert first._domain_values() == second._domain_values()

    def test_hiding_an_entity_removes_its_target_index(self):
        visible = _env()
        hidden = _env(entities=[{"h": "e2", "p": [-3.5, 0.5], "type": "container"}])
        assert "e1" in visible._domain_values()["target"]
        assert "e1" not in hidden._domain_values()["target"]

    def test_the_mask_narrows_when_the_scene_narrows(self):
        full = _env().action_masks().sum()
        bare = _env(entities=[], resources={"tiles": []}).action_masks().sum()
        assert bare < full


# ------------------------------------------------- the bounded smoke check


class _StubVecEnv(VecEnv):
    """Real MaskablePPO over the factorized space, no engine."""

    def __init__(self, template: ParameterizedEnv, num_envs: int = 2):
        self.template = template
        super().__init__(num_envs, template.observation_space, template.action_space)
        self.width = int(template.action_space.nvec.sum())
        self.taken: list[np.ndarray] = []
        self._actions = None

    def _obs(self):
        single = self.template.env._encoded()
        return {k: np.stack([v] * self.num_envs) for k, v in single.items()}

    def reset(self):
        return self._obs()

    def step_async(self, actions):
        self._actions = actions
        self.taken.append(np.asarray(actions).copy())

    def step_wait(self):
        return (
            self._obs(),
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=bool),
            [{} for _ in range(self.num_envs)],
        )

    def close(self):
        return None

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name == "action_masks":
            return [self.template.action_masks() for _ in range(self.num_envs)]
        raise NotImplementedError(method_name)

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        return None

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [seed] * self.num_envs

    def action_masks(self):
        return np.stack([self.template.action_masks() for _ in range(self.num_envs)])


def _model(vec, n_steps=8):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MultiInputPolicy",
        vec,
        n_steps=n_steps,
        batch_size=n_steps * vec.num_envs,
        n_epochs=1,
        device="cpu",
        verbose=0,
    )


class TestTheDistributionWorks:
    def test_sampling_log_probs_and_a_ppo_update(self):
        vec = _StubVecEnv(_env())
        model = _model(vec)
        updates = []
        real = model.train
        model.train = lambda *a, **k: (updates.append(model.num_timesteps), real())[1]
        model.learn(total_timesteps=32, progress_bar=False)
        assert updates, "no optimizer update ran on the factorized space"

    def test_no_sampled_action_lands_on_a_masked_index(self):
        vec = _StubVecEnv(_env())
        model = _model(vec)
        model.train = lambda *a, **k: None
        model.learn(total_timesteps=32, progress_bar=False)

        mask = vec.template.action_masks()
        nvec = [int(n) for n in vec.action_space.nvec]
        for batch in vec.taken:
            for row in np.asarray(batch).reshape(-1, len(nvec)):
                offset = 0
                for dim, size in enumerate(nvec):
                    assert mask[offset + int(row[dim])], (
                        f"dimension {dim} sampled a masked index {row[dim]}"
                    )
                    offset += size

    def test_log_probs_are_finite_and_the_mask_is_respected_by_the_distribution(self):
        space = spaces.MultiDiscrete([3, 4])
        dist = make_masked_proba_distribution(space)
        net = dist.proba_distribution_net(latent_dim=8)
        dist.proba_distribution(net(th.zeros(4, 8)))
        mask = np.ones((4, 7), dtype=bool)
        mask[:, 3:6] = False  # only the last value of dimension 1 is legal
        dist.apply_masking(mask)
        actions = dist.sample()
        assert (actions[:, 1] == 3).all()
        assert th.isfinite(dist.log_prob(actions)).all()

    def test_an_empty_candidate_set_still_yields_a_legal_sample(self):
        """Nothing visible: every argument dimension falls to the sentinel."""
        env = ParameterizedEnv(
            _Inner(_observation(entities=[], resources={"tiles": []}, inventory={}))
        )
        vec = _StubVecEnv(env)
        model = _model(vec, n_steps=4)
        model.train = lambda *a, **k: None
        model.learn(total_timesteps=8, progress_bar=False)
        assert vec.taken, "collection produced nothing"


def test_the_wrapper_does_not_change_the_discrete_path():
    """A flat checkpoint and a parameterized one differ only in actions the
    flat one never had -- the property `SkillEnv` established."""
    inner = _Inner(catalog="primitive-v1")
    before = len(inner.catalog)
    wrapped = ParameterizedEnv(inner)
    assert int(wrapped.action_space.nvec[0]) == before


@pytest.mark.parametrize("name", [name for name, _ in DIMENSIONS])
def test_every_dimension_is_named_and_sized(name):
    env = _env()
    assert name in env._sizes
    assert env._sizes[name] >= 1


class TestOperationsThatCanNeverDecodeAreMasked:
    """`craft_recipe`, `set_recipe_at` and `cancel_request` take a `recipe` or a
    `target_request_id`, and no dimension offers either. They were decode
    failures every time they were sampled, yet the operation mask was the
    catalog's own, so `craft_recipe` stayed legal whenever `recipes` was
    non-empty -- a free no-op the policy could learn to spend probability on."""

    def test_undecodable_operations_are_illegal_even_with_recipes_in_view(self):
        env = _env(recipes=[{"name": "iron-gear-wheel", "craftable": 3}])
        keys = env.env.catalog.keys()
        operations = env.action_masks()[: int(env.action_space.nvec[0])]
        for key in ("craft_recipe", "set_recipe_at", "cancel_request"):
            if key in keys:
                assert not operations[keys.index(key)], f"{key} can never decode"

    def test_every_legal_operation_is_decodable(self):
        env = _env(recipes=[{"name": "iron-gear-wheel", "craftable": 3}])
        operations = env.action_masks()[: int(env.action_space.nvec[0])]
        for index in np.flatnonzero(operations):
            assert env.decodable(int(index))

    def test_the_no_op_survives(self):
        env = _env(entities=[], resources={"tiles": []}, inventory={})
        operations = env.action_masks()[: int(env.action_space.nvec[0])]
        assert operations[env.env.catalog.wait_index]


class TestAnItemCanBeTakenThatIsNotYetHeld:
    """The item dimension offered only what the character held. `take_from`
    validates against `source_items` -- what visible entities hold -- so the
    first plate out of a furnace, or anything out of a chest the character had
    none of, could never be requested at all."""

    def _item_mask(self, env):
        mask = env.action_masks()
        offset = int(sum(env.action_space.nvec[:4]))
        item_mask = mask[offset : offset + int(env.action_space.nvec[4])]
        return {encoders.ITEMS[i - 1] for i in np.flatnonzero(item_mask) if i != UNUSED}

    def test_an_item_inside_a_visible_container_is_nameable(self):
        env = _env(
            entities=[
                {"h": "e2", "p": [-3.5, 0.5], "type": "container", "contents": {"iron-plate": 7}},
            ],
        )
        assert self._item_mask(env) == {"transport-belt", "iron-plate"}

    def test_furnace_output_is_nameable(self):
        env = _env(
            inventory={},
            entities=[
                {"h": "f1", "p": [1.5, 1.5], "type": "furnace", "output": {"iron-plate": 2}},
            ],
        )
        assert "iron-plate" in self._item_mask(env)


class TestOnePlaceDecidesTheActionSpace:
    """Training, evaluation, baselines and every vectorised worker wrapped the
    environment by hand, and each knew only about `SkillEnv`. A task on
    `parameterized-v1` therefore reached the learner as `Discrete(len(catalog))`,
    and stepping any template with a `?` argument through that path raises."""

    def test_a_parameterized_catalog_gets_the_factorized_space(self):
        from factoriorl.parameterized import wrap_for_policy

        wrapped = wrap_for_policy(_Inner())
        assert isinstance(wrapped, ParameterizedEnv)
        assert isinstance(wrapped.action_space, spaces.MultiDiscrete)

    def test_a_primitive_catalog_is_left_alone(self):
        from factoriorl.parameterized import wrap_for_policy

        inner = _Inner(catalog="primitive-v1")
        assert wrap_for_policy(inner) is inner

    def test_skills_over_a_parameterized_catalog_is_refused_not_ignored(self):
        from factoriorl.parameterized import wrap_for_policy

        with pytest.raises(ValueError, match="parameterized catalog"):
            wrap_for_policy(_Inner(), skills=True)

    def test_train_and_the_vectorised_workers_use_it(self):
        """A guard on the wiring itself: the bug was four hand-written copies."""
        import inspect

        from factoriorl import vecenv
        from factoriorl.learn import train

        assert "wrap_for_policy" in inspect.getsource(train._wrap)
        assert "wrap_for_policy(env, skills)" in inspect.getsource(vecenv)


class TestTheRandomFloorSamplesTheRealActionSpace:
    """`rng.choice(np.flatnonzero(mask))` is a catalog index for `Discrete` and
    nonsense for `MultiDiscrete`, where the flat mask is every dimension
    concatenated: the "random action" was an index into that concatenation."""

    def test_each_dimension_is_sampled_from_its_own_legal_values(self):
        from factoriorl.parameterized import sample_masked

        env = _env()
        rng = np.random.default_rng(0)
        mask = env.action_masks()
        for _ in range(200):
            action = sample_masked(env.action_space, mask, rng)
            assert action.shape == env.action_space.nvec.shape
            offset = 0
            for value, size in zip(action, env.action_space.nvec, strict=True):
                assert 0 <= value < size
                assert mask[offset + int(value)], "sampled a masked value"
                offset += int(size)

    def test_a_discrete_space_still_gets_a_catalog_index(self):
        from factoriorl.parameterized import sample_masked

        mask = np.array([False, True, False, True])
        rng = np.random.default_rng(0)
        drawn = {sample_masked(spaces.Discrete(4), mask, rng) for _ in range(50)}
        assert drawn == {1, 3}


class TestAPolicyActionReachesTheEnvironmentIntact:
    """`int(action)` was hard-coded where a learned policy's action meets an
    environment -- the vectorised worker and serial evaluation. The first real
    PPO run on a parameterized task (`build_line`) raised `TypeError: only
    0-dimensional arrays can be converted to Python scalars` on its first step.
    No stub test caught it because none drove `step_async`."""

    def test_a_factorized_vector_is_kept_whole(self):
        from factoriorl.parameterized import as_env_action

        space = spaces.MultiDiscrete([22, 33, 122, 5, 15, 4])
        action = as_env_action(space, np.array([12, 0, 7, 2, 0, 0]))
        assert isinstance(action, np.ndarray)
        assert action.tolist() == [12, 0, 7, 2, 0, 0]

    def test_a_discrete_index_is_still_an_int(self):
        from factoriorl.parameterized import as_env_action

        assert as_env_action(spaces.Discrete(23), np.array(7)) == 7
        assert as_env_action(spaces.Discrete(23), np.array([7])) == 7

    def test_the_vectorised_worker_submits_the_whole_vector(self):
        from concurrent.futures import ThreadPoolExecutor

        from factoriorl.vecenv import FactorioVecEnv

        received = []

        class _Worker:
            action_space = spaces.MultiDiscrete([22, 33, 122, 5, 15, 4])

            def step(self, action):
                received.append(action)
                return None

        vec = object.__new__(FactorioVecEnv)
        vec.envs = [_Worker(), _Worker()]
        vec._executor = ThreadPoolExecutor(max_workers=2)
        try:
            vec.step_async(np.array([[12, 0, 7, 2, 0, 0], [21, 0, 0, 0, 0, 0]]))
            for future in vec._pending:
                future.result()
        finally:
            vec._executor.shutdown()
        assert sorted(a.tolist() for a in received) == [
            [12, 0, 7, 2, 0, 0],
            [21, 0, 0, 0, 0, 0],
        ]
