"""Shared CPU, CUDA, and ACL backend selection helpers."""

import os


DEVICES = ("cpu", "cuda", "acl")


class BackendError(RuntimeError):
    """Raised when the selected execution backend cannot be used."""


def add_device_argument(parser, default="cuda"):
    """Add a common device selector to an argument parser."""
    parser.add_argument("--device", choices=DEVICES, default=default)
    return parser


def configure_device(device, jt_module=None):
    """Configure Jittor flags for the requested device backend."""
    _validate_device(device)
    jt_module = _jittor_module(jt_module)
    compiler = getattr(jt_module, "compiler", None)

    if device == "cuda" and not getattr(compiler, "has_cuda", False):
        raise BackendError("CUDA backend is unavailable")
    if device == "acl":
        if not getattr(compiler, "has_acl", False):
            raise BackendError("ACL backend is unavailable")
        if not hasattr(jt_module.flags, "use_acl"):
            raise BackendError("ACL backend is unavailable: Jittor has no use_acl flag")

    jt_module.flags.use_acl = int(device == "acl")
    jt_module.flags.use_cuda = int(device == "cuda")


def device_status(device, jt_module=None, environ=None):
    """Return backend, environment, compiler, and distributed-run state."""
    _validate_device(device)
    jt_module = _jittor_module(jt_module)
    compiler = getattr(jt_module, "compiler", None)
    flags = getattr(jt_module, "flags", None)
    environ = os.environ if environ is None else environ

    status = {
        "device": device,
        "jittor_commit": environ.get("NKAI_JITTOR_COMMIT"),
        "has_cuda": bool(getattr(compiler, "has_cuda", False)),
        "has_acl": bool(getattr(compiler, "has_acl", False)),
        "use_cuda": int(getattr(flags, "use_cuda", 0)),
        "use_acl": int(getattr(flags, "use_acl", 0)),
        "nvcc_path": getattr(compiler, "nvcc_path", None),
        "tikcc_path": getattr(compiler, "tikcc_path", None),
        "ascend_toolkit_home": environ.get("ASCEND_TOOLKIT_HOME"),
        "jittor_home": environ.get("JITTOR_HOME"),
        "in_mpi": bool(getattr(jt_module, "in_mpi", False)),
        "rank": getattr(jt_module, "rank", 0),
        "world_size": getattr(jt_module, "world_size", 1),
    }
    status.update(
        {
            "NKAI_JITTOR_COMMIT": status["jittor_commit"],
            "ASCEND_TOOLKIT_HOME": status["ascend_toolkit_home"],
            "JITTOR_HOME": status["jittor_home"],
        }
    )
    return status


def _validate_device(device):
    if device not in DEVICES:
        raise BackendError("Unknown device {!r}; expected one of {}".format(device, DEVICES))


def _jittor_module(jt_module):
    if jt_module is not None:
        return jt_module

    try:
        import jittor
    except ImportError as error:
        raise BackendError("Jittor is required to configure a device backend") from error
    return jittor
