from core.inference.factory import create_inference_engine
from core.processor.factory import create_processor
from core.communication.ros2_bridge import Ros2Bridge
from utils.message.message_convert import actions_dict_to_trajectory, get_action_time
from utils.action_trace import ActionTraceRecorder

from scheduler.instruction.instruction import InstructionManager, InstructionAction
from scheduler.trajectory.manager import TrajectoryManager, EnsembleMode
from utils.message.datatype import Trajectory, ExecutionMode
from std_msgs.msg import String
from loguru import logger

import toml
import torch
import time
import numpy as np
import copy
from omegaconf import OmegaConf
from pathlib import Path
from galaxea_fm.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

from accelerate import PartialState
distributed_state = PartialState()

class Scheduler:
    def __init__(self, config):
        self.schedule_config = config
        self.model_config = OmegaConf.load(f"{self.schedule_config['model']['ckpt_dir']}/config.yaml")
        self._rewrite_local_model_asset_paths()

        self.cnt = 0
        self.step_mode = self.schedule_config['basic']['step_mode']
        self.step_freq = self.schedule_config['basic']['control_frequency']
        self.num_of_steps = self.schedule_config['basic']['action_steps']
        self.last_instruction_text = ""
        self.last_user_instruction_text = ""
        self.last_model_instruction_text = ""
        self.action_trace = None

        self._setup_all()

    def run(self):
        while self.ros2_bridge.is_running():
            obs_time, obs = self.ros2_bridge.gather_obs()
            obs_state_for_trace = self._snapshot_obs_state(obs)
            object_context_for_trace = self._snapshot_object_context(obs)
            infer_start = time.time()
            actions = self.inference(obs)
            infer_cost = time.time() - infer_start
            if actions is not None and self.cnt >= 2:
                logger.info(f'Infer cost: {infer_cost}')
                debug_chunk_id = None
                if self.action_trace is not None:
                    debug_chunk_id = self.action_trace.record_predicted_chunk(
                        actions=actions["action"],
                        obs_time=obs_time,
                        infer_start=infer_start,
                        infer_end=infer_start + infer_cost,
                        instruction=self.last_user_instruction_text,
                        model_instruction=self.last_model_instruction_text,
                        obs_state=obs_state_for_trace,
                        object_context=object_context_for_trace,
                    )
                self.step(actions['action'], obs_time, debug_chunk_id=debug_chunk_id)
            self.cnt += 1

    def _snapshot_obs_state(self, obs):
        if not isinstance(obs, dict):
            return None
        raw_state = obs.get("state")
        if not isinstance(raw_state, dict):
            return raw_state

        snapshot = {}
        for key, value in raw_state.items():
            if value is None:
                continue
            if hasattr(value, "detach"):
                snapshot[key] = value.detach().clone()
            elif isinstance(value, np.ndarray):
                snapshot[key] = value.copy()
            else:
                snapshot[key] = copy.deepcopy(value)
        return snapshot

    def _snapshot_object_context(self, obs):
        trace_context = self.instruction_manager.get_trace_context()
        if not isinstance(obs, dict):
            return trace_context

        raw_images = obs.get("images")
        if not isinstance(raw_images, dict):
            return trace_context

        head_rgb = raw_images.get("head_rgb")
        if head_rgb is None:
            return trace_context

        if hasattr(head_rgb, "detach"):
            trace_context["head_rgb"] = head_rgb.detach().clone()
        elif isinstance(head_rgb, np.ndarray):
            trace_context["head_rgb"] = head_rgb.copy()
        else:
            trace_context["head_rgb"] = copy.deepcopy(head_rgb)
        return trace_context

    def inference(self, obs):
        if obs is None:
            if self.cnt % 100 == 0:
                logger.info("No observation")
            time.sleep(0.01)
            return

        instruct_action = self.instruction_manager.get_instruction(obs)
        if instruct_action == InstructionAction.RESET:
            self.ros2_bridge.reset()
            return
        elif instruct_action == InstructionAction.CONTINUE:
            pass
        elif instruct_action == InstructionAction.SKIP:
            return
        self.last_user_instruction_text = self.instruction_manager.last_instruction
        self.last_model_instruction_text = obs.get("task", self.instruction_manager.last_instruction)
        self.last_instruction_text = self.last_model_instruction_text

        batch = self.processor.preprocess(obs)
        for k, v in batch.items():
            if isinstance(v, str):
                batch[k] = [v]
            elif isinstance(v, bool):
                batch[k] = torch.tensor([v])
            else:
                batch[k] = v.unsqueeze(0)
        batch = self.inference_engine.predict_action(batch)
        batch["action"] = batch["action"].cpu()
        batch["proprio"] = batch["proprio"].cpu()
        actions = self.processor.postprocess(batch)
        return actions

    def step(self, actions: dict, obs_time: float, debug_chunk_id: int | None = None):
        if self.step_mode == "sync":
            trajectory = actions_dict_to_trajectory(actions=actions, time_step=1/self.step_freq, num_of_steps=self.num_of_steps, timestamp=self.ros2_bridge.now())
            if len(trajectory.actions) < self.num_of_steps:
                raise ValueError(f"Trajectory actions length {len(trajectory.actions)} is less than num_of_steps {self.num_of_steps}")

            self._sync_publish(trajectory, debug_chunk_id=debug_chunk_id)

        elif self.step_mode == "async":
            logger.info(f'Add actions to trajectory manager.')
            self.trajectory_manager.add_actions(actions, obs_time, debug_chunk_id=debug_chunk_id)
        else:
            raise ValueError(f"Invalid step mode: {self.step_mode}")

    def _sync_publish(self, trajectory: Trajectory, debug_chunk_id: int | None = None):
        for i in range(self.num_of_steps):
            self.ros2_bridge.publish_action(trajectory.actions[i])
            if self.action_trace is not None:
                self.action_trace.record_publish_event(
                    action=trajectory.actions[i],
                    publish_time=time.time(),
                    manager_debug={
                        "mode": "SYNC",
                        "status": "published",
                        "chunk_id": debug_chunk_id,
                        "primary_chunk_id": debug_chunk_id,
                        "step_index": i,
                        "scheduled_action_time": float(get_action_time(trajectory.actions[i])),
                    },
                    feedback_snapshot=self.ros2_bridge.get_latest_feedback_snapshot(),
                )
            time.sleep(1.0 / self.step_freq)

    @logger.catch
    def _async_publish(self):
        if not self.trajectory_manager.is_ready():
            return
        now = time.time()
        action = self.trajectory_manager.get_action(now)
        if action is None:
            if self.action_trace is not None:
                self.action_trace.record_publish_miss(
                    publish_time=now,
                    manager_debug=self.trajectory_manager.get_last_debug(),
                    feedback_snapshot=self.ros2_bridge.get_latest_feedback_snapshot(),
                )
            return
        self.ros2_bridge.publish_action(action)
        if self.action_trace is not None:
            self.action_trace.record_publish_event(
                action=action,
                publish_time=time.time(),
                manager_debug=self.trajectory_manager.get_last_debug(),
                feedback_snapshot=self.ros2_bridge.get_latest_feedback_snapshot(),
            )

    def _setup_all(self):
        self._setup_processor()
        self._setup_trajectory_manager()
        self._setup_instruction_manager()
        self._setup_ros2_bridge()
        self._setup_inference_engine()
        self._setup_action_trace()

    def _rewrite_local_model_asset_paths(self):
        ckpt_dir = Path(self.schedule_config["model"]["ckpt_dir"]).resolve()
        candidate_google_dirs = [
            ckpt_dir.parent / "google",
            ckpt_dir.parent.parent / "google",
            Path.home() / "g0plus_ros2" / "data" / "google",
        ]

        local_google_dir = next((path for path in candidate_google_dirs if path.exists()), None)
        if local_google_dir is None:
            return

        replacements = [
            ("model.processor.tokenizer_params.pretrained_model_name_or_path", str(local_google_dir)),
            ("model.model_arch.pretrained_model_path", str(local_google_dir)),
        ]

        for key_path, replacement in replacements:
            current_value = OmegaConf.select(self.model_config, key_path)
            if isinstance(current_value, str) and current_value.startswith("/data/google/"):
                OmegaConf.update(self.model_config, key_path, replacement, merge=False)
                logger.info(f"Rewrote model asset path {key_path} -> {replacement}")

    def _setup_inference_engine(self):
        self.inference_engine = create_inference_engine(self.schedule_config, self.model_config, use_trt=self.schedule_config['model']['use_trt'])
        self.inference_engine.load_model()

    def _setup_processor(self):
        self.processor = create_processor(self.schedule_config, self.model_config, processor_type=self.schedule_config['model']['processor'])
        self.processor.initialize(Path(f"{self.schedule_config['model']['ckpt_dir']}/dataset_stats.json"))

    def _setup_trajectory_manager(self):
        if self.schedule_config['trajectory']['ensemble_mode'] == "RTC":
            ensemble_mode = EnsembleMode.RTC
        elif self.schedule_config['trajectory']['ensemble_mode'] == "RTG":
            ensemble_mode = EnsembleMode.RTG
        elif self.schedule_config['trajectory']['ensemble_mode'] == "HATO":
            ensemble_mode = EnsembleMode.HATO
        else:
            logger.warning(f"Invalid ensemble mode:{self.schedule_config['trajectory']['ensemble_mode']}")
            ensemble_mode = EnsembleMode.NONE
        
        if self.schedule_config['trajectory']['execution_mode'] == "JOINT_STATE":
            execution_mode = ExecutionMode.JOINT_STATE
        elif self.schedule_config['trajectory']['execution_mode'] == "EE_POSE":
            execution_mode = ExecutionMode.EE_POSE
        else:
            raise ValueError(f"Invalid execution mode: {self.schedule_config['trajectory']['execution_mode']}")
        
        self.trajectory_manager = TrajectoryManager(
            ensemble_mode=ensemble_mode,
            execution_mode=execution_mode,
            dt=1 / self.step_freq,
        )
        self.trajectory_manager.start()

    def _setup_instruction_manager(self):
        self.instruction_manager = InstructionManager(self.schedule_config["instruction"])

    def _setup_ros2_bridge(self):
        # HACK: use_recv_time=True to use the received time from ROS2 messages
        self.ros2_bridge = Ros2Bridge(self.schedule_config, self.model_config, use_recv_time=True)
        self.ros2_bridge.register_subscription(String, 'hs/vlm_out2vla', self.instruction_manager._ehi_instruction_callback)
        
        if self.step_mode == "async":
            self.ros2_bridge.register_publish_callback(self.step_freq, self._async_publish)

    def _setup_action_trace(self):
        visualization_cfg = self.schedule_config.get("visualization", {})
        if not visualization_cfg.get("enabled", True):
            return

        output_dir = visualization_cfg.get("output_dir")
        if output_dir is None or output_dir == "":
            output_dir = Path(self.schedule_config["model"]["ckpt_dir"]) / "efmnode_action_viz"

        self.action_trace = ActionTraceRecorder(
            output_dir=Path(output_dir),
            execution_mode=self.trajectory_manager.execution_mode,
            control_frequency=self.step_freq,
            action_steps=self.num_of_steps,
            render_every_n_publishes=visualization_cfg.get("render_every_n_publishes", 5),
            keep_chunks=visualization_cfg.get("keep_chunks", 12),
            keep_publish_events=visualization_cfg.get("keep_publish_events", 400),
        )
 

if __name__ == "__main__":
    config = toml.load("config.toml")
    scheduler = Scheduler(config)
    scheduler.run()
