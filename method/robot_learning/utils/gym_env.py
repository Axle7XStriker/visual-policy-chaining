from collections import deque, OrderedDict

import gym
import numpy as np


def cat_spaces(spaces):
    if isinstance(spaces[0], gym.spaces.Box):
        out_space = gym.spaces.Box(
            low=np.concatenate([s.low for s in spaces]),
            high=np.concatenate([s.high for s in spaces]),
        )
    elif isinstance(spaces[0], gym.spaces.Discrete):
        out_space = gym.spaces.Discrete(sum([s.n for s in spaces]))
    return out_space


def stacked_space(space, k):
    if isinstance(space, gym.spaces.Box):
        space_stack = gym.spaces.Box(
            low=np.concatenate([space.low] * k, axis=0),
            high=np.concatenate([space.high] * k, axis=0),
        )
    elif isinstance(space, gym.spaces.Discrete):
        space_stack = gym.spaces.Discrete(space.n * k)
    return space_stack


def value_to_space(value):
    if isinstance(value, dict):
        space = gym.spaces.Dict(
            [(k, value_to_space(v)) for k, v in value.items()]
        )
    elif isinstance(value, np.ndarray):
        space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=value.shape)
    else:
        raise NotImplementedError

    return space


def spaces_to_shapes(space):
    if isinstance(space, gym.spaces.Dict):
        return {k: spaces_to_shapes(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Box):
        return space.shape
    elif isinstance(space, gym.spaces.Discrete):
        return [space.n]


def zero_value(space, dtype=np.float64):
    if isinstance(space, gym.spaces.Dict):
        return OrderedDict(
            [(k, zero_value(space, dtype)) for k, space in space.spaces.items()]
        )
    elif isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape).astype(dtype)
    elif isinstance(space, gym.spaces.Discrete):
        return np.zeros(1).astype(dtype)


def get_non_absorbing_state(ob):
    ob = ob.copy()
    ob["absorbing_state"] = np.array([0])
    return ob


def get_absorbing_state(space):
    ob = zero_value(space)
    ob["absorbing_state"] = np.array([1])
    return ob


class GymWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        from_pixels=False,
        height=100,
        width=100,
        camera_id=None,
        channels_first=True,
        frame_skip=1,
        return_state=False,
    ):
        super().__init__(env)
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._channels_first = channels_first
        self._frame_skip = frame_skip
        # self.max_episode_steps = self.env._max_episode_steps // frame_skip
        if self.env.spec.max_episode_steps:
            max_episode_steps = self.env.spec.max_episode_steps
        else:
            max_episode_steps = self.env.spec.kwargs["max_episode_steps"]
        self.max_episode_steps = max_episode_steps // frame_skip

        self._return_state = return_state

        ob_space = {}
        for k, v in env.observation_space.spaces.items():
            if from_pixels and len(v.shape) == 3:
                shape = [1, height, width] if channels_first else [height, width, 3]
            elif from_pixels and len(v.shape) == 4:
                shape = [v.shape[0], 1, height, width] if channels_first else [v.shape[0], height, width, 1]
                if v.shape[0] == 1:
                    shape.remove(v.shape[0])

            if len(v.shape) in [3, 4]:
                ob_space[k] = gym.spaces.Box(
                    low=0, high=255, shape=shape, dtype=np.uint8
                )
            else:
                ob_space[k] = v
        self.observation_space = gym.spaces.Dict(ob_space)
        self.env_observation_space = env.observation_space

    def reset(self):
        ob = self.env.reset()

        if self._return_state:
            return self._get_obs(ob, reset=True), ob

        return self._get_obs(ob, reset=True)

    def step(self, ac):
        reward = 0
        for _ in range(self._frame_skip):
            ob, _reward, done, info = self.env.step(ac)
            reward += _reward
            if done:
                break
        if self._return_state:
            return (self._get_obs(ob), ob), reward, done, info

        return self._get_obs(ob), reward, done, info

    def _get_obs(self, ob, reset=False):
        # if self._from_pixels:
        #     ob = self.render(
        #         mode="rgb_array",
        #         # height=self._height,
        #         # width=self._width,
        #         # camera_id=self._camera_id,
        #     )
        #     if reset:
        #         ob = self.render(
        #             mode="rgb_array",
        #             # height=self._height,
        #             # width=self._width,
        #             # camera_id=self._camera_id,
        #         )
        #     if self._channels_first:
        #         ob = ob[0].transpose(2, 0, 1).copy()
        for k, v in ob.items():
            if len(v.shape) == 3:
                ob[k] = ob[k].transpose(2, 0, 1).copy() if self._channels_first else ob[k].copy()
            elif len(v.shape) == 4:
                ob[k] = ob[k].transpose(0, 3, 1, 2).copy() if self._channels_first else ob[k].copy()
                if v.shape[0] == 1:
                    ob[k] = ob[k].squeeze(0)

        return ob


class DictWrapper(gym.Wrapper):
    def __init__(self, env, return_state=False):
        super().__init__(env)

        self._return_state = return_state

        self._is_ob_dict = isinstance(env.observation_space, gym.spaces.Dict)
        if not self._is_ob_dict:
            self.observation_space = gym.spaces.Dict({"ob": env.observation_space})
            self.env_observation_space = gym.spaces.Dict(
                {"state": env.env_observation_space}
            )
        else:
            self.observation_space = env.observation_space
            self.env_observation_space = env.env_observation_space

        self._is_ac_dict = isinstance(env.action_space, gym.spaces.Dict)
        if not self._is_ac_dict:
            self.action_space = gym.spaces.Dict({"ac": env.action_space})
        else:
            self.action_space = env.action_space

    def reset(self):
        ob = self.env.reset()
        return self._get_obs(ob)

    def step(self, ac):
        if not self._is_ac_dict:
            ac = ac["ac"]
        ob, reward, done, info = self.env.step(ac)
        return self._get_obs(ob), reward, done, info

    def _get_obs(self, ob):
        if not self._is_ob_dict:
            if self._return_state:
                ob = {"ob": ob[0], "state": ob[1]}
            else:
                ob = {"ob": ob}
        return ob


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack=3, return_state=False):
        super().__init__(env)

        # Both observation and action spaces must be gym.spaces.Dict.
        assert isinstance(env.observation_space, gym.spaces.Dict), env.observation_space
        assert isinstance(env.action_space, gym.spaces.Dict), env.action_space

        self._frame_stack = frame_stack
        self._frames = deque([], maxlen=frame_stack)
        self._return_state = return_state
        self._state = None

        ob_space = []
        for k, space in env.observation_space.spaces.items():
            space_stack = stacked_space(space, frame_stack)
            ob_space.append((k, space_stack))
        self.observation_space = gym.spaces.Dict(ob_space)

        self.env_observation_space = env.env_observation_space

    def reset(self):
        ob = self.env.reset()
        if self._return_state:
            self._state = ob.pop("state", None)
        for _ in range(self._frame_stack):
            self._frames.append(ob)
        return self._get_obs()

    def step(self, ac):
        ob, reward, done, info = self.env.step(ac)
        if self._return_state:
            self._state = ob.pop("state", None)
        self._frames.append(ob)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        frames = list(self._frames)
        obs = []
        for k in self.env.observation_space.spaces.keys():
            obs.append((k, np.concatenate([f[k] for f in frames], axis=0)))
        if self._return_state:
            obs.append(("state", self._state))

        return OrderedDict(obs)


class AbsorbingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        ob_space = gym.spaces.Dict(spaces=dict(env.observation_space.spaces))
        ob_space.spaces["absorbing_state"] = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.uint8
        )
        self.observation_space = ob_space

    def reset(self):
        ob = self.env.reset()
        return self._get_obs(ob)

    def step(self, ac):
        ob, reward, done, info = self.env.step(ac)
        return self._get_obs(ob), reward, done, info

    def _get_obs(self, ob):
        return get_non_absorbing_state(ob)

    def get_absorbing_state(self):
        return get_absorbing_state(self.observation_space)
