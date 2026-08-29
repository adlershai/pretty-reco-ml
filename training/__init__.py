"""Two-tower training package.

Loss: in-batch sampled softmax (InfoNCE). In-batch comparisons are a training
mechanism, not historical “customer rejected this model” labels.

Lifecycle (not scheduled in this phase):
- new/updated customer → Customer Tower only
- new model → Model Tower only
- end of season / ~3–4k new purchases → retrain both towers, promote if better
"""

from training.artifact import load_artifact
from training.two_tower import TwoTowerModel

__all__ = ["TwoTowerModel", "load_artifact"]
