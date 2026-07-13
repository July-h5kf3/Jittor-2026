import os
from typing import Mapping, Optional


def configure_jittor_runtime(jt_module, environ: Optional[Mapping[str, str]] = None) -> None:
    env = os.environ if environ is None else environ
    jt_module.flags.use_cuda = int(env.get("JITTOR_USE_CUDA", "1"))
