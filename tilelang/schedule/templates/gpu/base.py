"""Base schedule rule for GPU operators."""

from tilelang import tvm

from tvm.target import Target
from tvm.dlight import ScheduleRule

class GPUScheduleRule(ScheduleRule):  # pylint: disable=too-few-public-methods
    """The Schedule Rule specific to GPU targets, will return None if the target is not GPU."""

    def is_target_available(self, target: Target) -> bool:
        """Check whether the target is available for gpu rule.

        Parameters
        ----------
        target : Target
            The compilation target to check.

        Returns
        -------
        available : bool
            Whether the target is available for this rule.
        """
        return super().is_target_available(target) and "gpu" in target.keys
