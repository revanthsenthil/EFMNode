from core.communication.message_queue import MessageQueue
from utils.message.datatype import RobotAction, Trajectory
from utils.message.message_convert import actions_dict_to_trajectory, actions_dict_to_array, array_to_action, get_action_time
from scheduler.trajectory.stitcher import TrajectoryStitcher
from utils.message.datatype import ExecutionMode
from enum import Enum
from typing import Optional, Literal
import threading
import time
from loguru import logger
import numpy as np
from scheduler.trajectory.hato import ensemble
import copy
class EnsembleMode(Enum):
    NONE = "NONE"
    HATO = "HATO"
    RTG = "RTG"
    RTC = "RTC"

MAX_ACTIONS_QUEUE_LENGTH = {
    EnsembleMode.NONE: 1,
    EnsembleMode.HATO: 4,
    EnsembleMode.RTG: 1,
    EnsembleMode.RTC: 1,
}

class TrajectoryManager:
    def __init__(self, ensemble_mode: Optional[EnsembleMode] = EnsembleMode.NONE, execution_mode: Optional[ExecutionMode] = ExecutionMode.JOINT_STATE, dt: float=1/15):
        self.ensemble_mode = ensemble_mode
        self.execution_mode = execution_mode
        logger.info(f"Ensemble mode: {ensemble_mode}, execution mode: {execution_mode}")
        self.max_actions_queue_length = MAX_ACTIONS_QUEUE_LENGTH[ensemble_mode]
        self.actions_queue = MessageQueue(maxlen=self.max_actions_queue_length)
        
        self.trajectory = None
        self.stitcher = TrajectoryStitcher(execution_mode=self.execution_mode)

        self.is_actions_queue_updated = False
        # self.traj_worker_stop_event = threading.Event()
        # self.traj_worker = threading.Thread(target=self._traj_worker, daemon=True)
        self.action = None
        self.dt = dt
        self.mode = "next"
        self.active_chunk_id = None
        self.active_step_idx = 0
        self.last_action_debug = {"mode": ensemble_mode.value, "status": "init"}

    def start(self):
        pass
        # self.traj_worker.start()

    def stop(self):
        pass
        # self.traj_worker_stop_event.set()
        # self.traj_worker.join()

    def __del__(self):
        self.stop()

    def add_actions(self, actions: dict, obs_time: float = None, debug_chunk_id: int | None = None):
        logger.info(f'OBS: {obs_time}')
        obs_actions_dict = {
            "obs_time": obs_time,
            "actions": actions,
            "chunk_id": debug_chunk_id,
        }
        self.actions_queue.append(obs_actions_dict)
        if self.ensemble_mode in [EnsembleMode.NONE, EnsembleMode.RTC]:
            self._generate_trajectory(timestamp=obs_time)
        self.is_actions_queue_updated = True

    def get_action(self, timestamp: float = None) -> RobotAction:
        if self.ensemble_mode == EnsembleMode.HATO:
            ensembled_action = self._generate_trajectory(timestamp=timestamp)
            return ensembled_action
        elif self.ensemble_mode == EnsembleMode.RTG:
            if self.trajectory is None or len(self.trajectory.actions) == 0:
                self.last_action_debug = {
                    "mode": self.ensemble_mode.value,
                    "status": "no_action",
                    "reason": "empty_trajectory",
                    "timestamp": float(timestamp) if timestamp is not None else None,
                }
                return None
            self.action = self.trajectory.actions.popleft()
            self.last_action_debug = {
                "mode": self.ensemble_mode.value,
                "status": "published",
                "chunk_id": self.active_chunk_id,
                "primary_chunk_id": self.active_chunk_id,
                "step_index": self.active_step_idx,
                "timestamp": float(timestamp) if timestamp is not None else None,
                "scheduled_action_time": float(get_action_time(self.action)),
            }
            self.active_step_idx += 1
            return self.action
        elif self.ensemble_mode == EnsembleMode.NONE:
            if self.trajectory is None or len(self.trajectory.actions) == 0:
                self.last_action_debug = {
                    "mode": self.ensemble_mode.value,
                    "status": "no_action",
                    "reason": "empty_trajectory",
                    "timestamp": float(timestamp) if timestamp is not None else None,
                }
                return None
            actions_length = len(self.trajectory.actions)
            for _ in range(actions_length):
                action = self.trajectory.actions.popleft()
                action_time = get_action_time(action)
                step_index = self.active_step_idx
                self.active_step_idx += 1
                if timestamp < action_time:
                    self.last_action_debug = {
                        "mode": self.ensemble_mode.value,
                        "status": "published",
                        "chunk_id": self.active_chunk_id,
                        "primary_chunk_id": self.active_chunk_id,
                        "step_index": step_index,
                        "timestamp": float(timestamp) if timestamp is not None else None,
                        "scheduled_action_time": float(action_time),
                    }
                    return action
            self.last_action_debug = {
                "mode": self.ensemble_mode.value,
                "status": "no_action",
                "reason": "no_future_action_in_trajectory",
                "chunk_id": self.active_chunk_id,
                "timestamp": float(timestamp) if timestamp is not None else None,
            }
        elif self.ensemble_mode == EnsembleMode.RTC:
            if self.trajectory is None or len(self.trajectory.actions) == 0:
                self.last_action_debug = {
                    "mode": self.ensemble_mode.value,
                    "status": "no_action",
                    "reason": "empty_trajectory",
                    "timestamp": float(timestamp) if timestamp is not None else None,
                }
                return None
            actions_length = len(self.trajectory.actions)
            for _ in range(actions_length):
                action = self.trajectory.actions.popleft()
                action_time = get_action_time(action)
                step_index = self.active_step_idx
                self.active_step_idx += 1
                if timestamp < action_time:
                    self.last_action_debug = {
                        "mode": self.ensemble_mode.value,
                        "status": "published",
                        "chunk_id": self.active_chunk_id,
                        "primary_chunk_id": self.active_chunk_id,
                        "step_index": step_index,
                        "timestamp": float(timestamp) if timestamp is not None else None,
                        "scheduled_action_time": float(action_time),
                    }
                    return action
            self.last_action_debug = {
                "mode": self.ensemble_mode.value,
                "status": "no_action",
                "reason": "no_future_action_in_trajectory",
                "chunk_id": self.active_chunk_id,
                "timestamp": float(timestamp) if timestamp is not None else None,
            }
        else:
            raise NotImplementedError

    def get_last_action(self, timestamp: float = None):
        return self.action

    def get_last_debug(self) -> dict:
        return copy.deepcopy(self.last_action_debug)
    
    def is_ready(self):
        if self.ensemble_mode == EnsembleMode.RTG:
            return self.trajectory is not None and len(self.trajectory.actions) > 0
        else:
            return self.is_actions_queue_updated

    def _generate_trajectory(self, timestamp: float = None) -> Trajectory:
        if self.ensemble_mode == EnsembleMode.NONE:
            if len(self.actions_queue) == 0:
                self.last_action_debug = {
                    "mode": self.ensemble_mode.value,
                    "status": "no_action",
                    "reason": "empty_actions_queue",
                    "timestamp": float(timestamp) if timestamp is not None else None,
                }
                return None
            queued_actions = self.actions_queue.popleft()
            actions = queued_actions["actions"]
            self.active_chunk_id = queued_actions.get("chunk_id")
            self.active_step_idx = 0
            self.trajectory = actions_dict_to_trajectory(actions=actions, timestamp=timestamp)
        elif self.ensemble_mode == EnsembleMode.HATO:
            idxs = []
            raw_actions = []
            selected_chunk_ids = []
            selected_step_times = []
            for queued_actions in self.actions_queue:
                obs_time = queued_actions["obs_time"]
                actions = actions_dict_to_array(queued_actions["actions"], self.execution_mode)[0]
                actions_time = obs_time + self.dt * np.arange(actions.shape[0])
                mask = actions_time > timestamp
                if not np.any(mask):
                    continue
                idx = int(np.argmax(mask))
                idxs.append(idx)
                selected_chunk_ids.append(queued_actions.get("chunk_id"))
                selected_step_times.append(float(actions_time[idx]))
                if self.mode == "next":
                    raw_actions.append(actions[idx])
                elif self.mode == "interp":
                    raw_action = np.interp(actions_time, [actions_time[idx - 1], actions_time[idx]], [actions[idx - 1], actions[idx]])
                    raw_actions.append(raw_action)
                else:
                    raise NotImplementedError
            logger.info(f'use {idxs}')
            
            if len(raw_actions) == 0:
                self.last_action_debug = {
                    "mode": self.ensemble_mode.value,
                    "status": "no_action",
                    "reason": "no_future_steps_available",
                    "timestamp": float(timestamp) if timestamp is not None else None,
                    "queue_size": len(self.actions_queue),
                    "queue_chunk_ids": [queued.get("chunk_id") for queued in self.actions_queue],
                }
                return None

            ensembled_action = ensemble(raw_actions, self.execution_mode)
            weights = 0.5 ** np.arange(len(raw_actions) - 1, -1, -1)
            weights = weights / weights.sum()
            self.last_action_debug = {
                "mode": self.ensemble_mode.value,
                "status": "published",
                "timestamp": float(timestamp) if timestamp is not None else None,
                "queue_size": len(self.actions_queue),
                "selected_chunk_ids": list(selected_chunk_ids),
                "selected_step_indices": list(idxs),
                "selected_step_times": list(selected_step_times),
                "primary_chunk_id": selected_chunk_ids[-1] if selected_chunk_ids else None,
                "weights": weights.tolist(),
                "ensembled_action": np.asarray(ensembled_action, dtype=np.float32).tolist(),
            }

            return array_to_action(ensembled_action, self.execution_mode, timestamp=timestamp)

        elif self.ensemble_mode == EnsembleMode.RTG:
            if self.trajectory is None or len(self.trajectory.actions) == 0:
                queued_actions = self.actions_queue.popleft()
                self.active_chunk_id = queued_actions.get("chunk_id")
                self.active_step_idx = 0
                self.trajectory = actions_dict_to_trajectory(actions=queued_actions["actions"], timestamp=timestamp)
            elif len(self.trajectory.actions) < 16:
                queued_actions = self.actions_queue.popleft()
                self.active_chunk_id = queued_actions.get("chunk_id")
                self.active_step_idx = 0
                new_traj = actions_dict_to_trajectory(actions=queued_actions["actions"], timestamp=timestamp)
                self.trajectory = self.stitcher.stitch(self.trajectory, new_traj)
        elif self.ensemble_mode == EnsembleMode.RTC:
            if len(self.actions_queue) == 0:
                self.last_action_debug = {
                    "mode": self.ensemble_mode.value,
                    "status": "no_action",
                    "reason": "empty_actions_queue",
                    "timestamp": float(timestamp) if timestamp is not None else None,
                }
                return None
        else:
            logger.info(f'Not implement')

    # def _traj_worker(self):
    #     while not self.traj_worker_stop_event.is_set():
    #         if self.is_actions_queue_updated:
    #             self._generate_trajectory(timestamp=time.time())
    #             self.is_actions_queue_updated = False
    #         time.sleep(0.01)
