from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.assets import Articulation
from isaaclab.utils import configclass

class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action
    
class PostStepTorqueRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_post_step(self):
        asset: Articulation = self._env.scene["robot"]
        return "torques", asset.data.applied_torque

class PreStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_pre_step(self):
        return "states", self._env.scene.get_state(is_relative=True)

@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = PreStepActionsRecorder

@configclass
class PostStepTorqueRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = PostStepTorqueRecorder

@configclass
class PreStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = PreStepStatesRecorder