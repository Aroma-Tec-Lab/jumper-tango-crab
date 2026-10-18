"""jumper.dance_tango -- a tango solo (caminata, ochos, boleo, giro, quebrada), learned by imitation.

The same environment as every `jumper.dance*` task (`tasks/jumper/common/dance/env.py`);
this directory adds the clip and the numbers. The clip, `media/tango.npz`, is generated from
`choreo/choreography_tango.py` by `choreo/generate_tango.py`; see `env_cfg.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_tango",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed tango solo (caminata, ochos, boleo, giro, "
                "corte y quebrada); the clip is committed in tasks/jumper/dance_tango/media/",
    tags=("imitation", "jumper", "dance", "flat", "balance"),
)
