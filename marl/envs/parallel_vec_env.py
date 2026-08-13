"""Windows-safe project VecEnv with child errors and state RPCs.
The public reset/step contract matches HARL's ShareSubprocVecEnv.  This
project-owned variant adds the post-update state operations required for exact
training resume and never writes logs/checkpoints from child processes.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import traceback
from typing import Any, Callable, Sequence

import numpy as np


class RemoteWorkerError(RuntimeError):
    """A child environment failed during construction/reset/step/state RPC."""


def _worker(remote: Any, parent_remote: Any, env_fn: Callable[[], Any]) -> None:
    parent_remote.close()
    env = None
    try:
        env = env_fn()
        remote.send(("ok", None))
        while True:
            command, data = remote.recv()
            try:
                if command == "get_spec":
                    result = {
                        "observation_space": env.observation_space,
                        "share_observation_space": env.share_observation_space,
                        "action_space": env.action_space,
                        "n_agents": int(env.n_agents),
                        "agent_order": tuple(env.agent_order),
                    }
                elif command == "reset":
                    result = env.reset()
                elif command == "step":
                    ob, state, reward, done, info, available_actions = env.step(data)
                    if bool(np.all(done)):
                        info[0]["original_obs"] = copy.deepcopy(ob)
                        info[0]["original_state"] = copy.deepcopy(state)
                        info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                        ob, state, available_actions = env.reset()
                    result = (ob, state, reward, done, info, available_actions)
                elif command == "get_state":
                    from marl.checkpointing.environment_state import single_environment_state_dict

                    result = single_environment_state_dict(env)
                elif command == "set_state":
                    from marl.checkpointing.environment_state import load_single_environment_state_dict

                    load_single_environment_state_dict(env, data)
                    result = None
                elif command == "close":
                    env.close()
                    remote.send(("ok", None))
                    break
                else:
                    raise NotImplementedError(f"Unsupported child command {command!r}.")
                remote.send(("ok", result))
            except BaseException as exc:
                remote.send(
                    (
                        "error",
                        {
                            "command": command,
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                )
                try:
                    env.close()
                finally:
                    break
    except BaseException as exc:
        try:
            remote.send(
                (
                    "error",
                    {
                        "command": "construct",
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            remote.close()
        except OSError:
            pass


class ProjectShareSubprocVecEnv:
    """True spawn-based subprocess VecEnv for two/four formal rollout workers."""

    agent_order = ("idc", "bess")
    vec_env_type = "ProjectShareSubprocVecEnv"
    multiprocessing_start_method = "spawn"

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        *,
        worker_seeds: Sequence[int],
    ) -> None:
        if len(env_fns) < 2:
            raise ValueError("ProjectShareSubprocVecEnv requires at least two workers.")
        if len(env_fns) != len(worker_seeds):
            raise ValueError("env_fns and worker_seeds must have equal length.")
        self.num_envs = len(env_fns)
        self.worker_seeds = tuple(int(seed) for seed in worker_seeds)
        self.waiting = False
        self.closed = False
        self._failed = False
        context = mp.get_context(self.multiprocessing_start_method)
        pairs = [context.Pipe() for _ in env_fns]
        self.remotes, work_remotes = zip(*pairs)
        self.ps = [
            context.Process(target=_worker, args=(work, remote, env_fn), daemon=True)
            for work, remote, env_fn in zip(work_remotes, self.remotes, env_fns, strict=True)
        ]
        try:
            for process in self.ps:
                process.start()
            for work in work_remotes:
                work.close()
            self._receive_all("construct")
            specs = self._request_all("get_spec", None)
            first = specs[0]
            for rank, spec in enumerate(specs[1:], start=1):
                if (
                    spec["n_agents"] != first["n_agents"]
                    or spec["agent_order"] != first["agent_order"]
                    or spec["observation_space"] != first["observation_space"]
                    or spec["share_observation_space"] != first["share_observation_space"]
                    or spec["action_space"] != first["action_space"]
                ):
                    raise RuntimeError(f"Worker {rank} environment spaces differ from worker 0.")
            self.n_agents = int(first["n_agents"])
            self.agent_order = tuple(first["agent_order"])
            self.observation_space = first["observation_space"]
            self.share_observation_space = first["share_observation_space"]
            self.action_space = first["action_space"]
        except BaseException:
            self._failed = True
            self.close(force=True)
            raise

    @property
    def process_ids(self) -> tuple[int | None, ...]:
        return tuple(process.pid for process in self.ps)

    @property
    def alive(self) -> tuple[bool, ...]:
        return tuple(process.is_alive() for process in self.ps)

    def reset(self):
        results = self._request_all("reset", None)
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def step_async(self, actions: Any) -> None:
        if self.waiting:
            raise RuntimeError("A subprocess VecEnv step is already pending.")
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} worker actions, got {len(actions)}.")
        for remote, action in zip(self.remotes, actions, strict=True):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        if not self.waiting:
            raise RuntimeError("step_wait() called without step_async().")
        try:
            results = self._receive_all("step")
        finally:
            self.waiting = False
        obs, share_obs, rewards, dones, infos, available_actions = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rewards),
            np.stack(dones),
            tuple(infos),
            np.stack(available_actions),
        )

    def step(self, actions: Any):
        self.step_async(actions)
        return self.step_wait()

    def get_env_states(self) -> list[dict[str, Any]]:
        return self._request_all("get_state", None)

    def set_env_states(self, states: Sequence[dict[str, Any]]) -> None:
        if len(states) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} worker states, got {len(states)}.")
        for remote, state in zip(self.remotes, states, strict=True):
            remote.send(("set_state", state))
        self._receive_all("set_state")

    def close(self, *, force: bool = False) -> None:
        if self.closed:
            return
        if force or self._failed:
            for process in self.ps:
                if process.is_alive():
                    process.terminate()
        else:
            try:
                if self.waiting:
                    self._receive_all("step")
                    self.waiting = False
                for remote, process in zip(self.remotes, self.ps, strict=True):
                    if process.is_alive():
                        remote.send(("close", None))
                for remote, process in zip(self.remotes, self.ps, strict=True):
                    if process.is_alive():
                        status, payload = remote.recv()
                        if status == "error":
                            self._failed = True
            except (BrokenPipeError, EOFError, OSError):
                self._failed = True
        for process in self.ps:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for remote in self.remotes:
            try:
                remote.close()
            except OSError:
                pass
        self.closed = True

    def _request_all(self, command: str, payload: Any) -> list[Any]:
        for remote in self.remotes:
            remote.send((command, payload))
        return self._receive_all(command)

    def _receive_all(self, command: str) -> list[Any]:
        results = []
        errors = []
        for rank, remote in enumerate(self.remotes):
            try:
                status, payload = remote.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                errors.append(
                    {"rank": rank, "command": command, "type": type(exc).__name__, "message": str(exc), "traceback": ""}
                )
                continue
            if status == "error":
                payload = dict(payload)
                payload["rank"] = rank
                errors.append(payload)
            else:
                results.append(payload)
        if errors:
            self._failed = True
            details = "\n\n".join(
                f"worker={item['rank']} command={item['command']} {item['type']}: {item['message']}\n{item['traceback']}"
                for item in errors
            )
            raise RemoteWorkerError("Subprocess environment failure:\n" + details)
        return results

    def __del__(self) -> None:
        try:
            self.close(force=True)
        except Exception:
            pass
