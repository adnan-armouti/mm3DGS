from .math import (
    perp_stark, to_local, to_global,
    sample_cosine_hemisphere_concentric, sample_uniform_hemisphere,
    safe_normalize, gather_point3f, gather_vector3f,
    check_visibility_bidirectional,
)
from .transforms import euler_to_quaternion, quaternion_rotate, transform_positions, create_pose_params
from .optimizer import DrJitAdam
