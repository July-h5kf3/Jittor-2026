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
            raise BackendError("Jittor does not expose the ACL device flag")

    flags = jt_module.flags
    if device == "acl":
        flags.use_cuda = 0
        flags.use_acl = 1
    elif device == "cuda":
        if hasattr(flags, "use_acl"):
            flags.use_acl = 0
        flags.use_cuda = 1
    else:
        flags.use_cuda = 0
        if hasattr(flags, "use_acl"):
            flags.use_acl = 0


def device_status(device, jt_module=None, environ=None):
    """Return backend, environment, compiler, and distributed-run state."""
    _validate_device(device)
    jt_module = _jittor_module(jt_module)
    compiler = getattr(jt_module, "compiler", None)
    flags = getattr(jt_module, "flags", None)
    environ = os.environ if environ is None else environ

    status = {
        "device": device,
        "jittor_commit": environ.get("NKAI_JITTOR_COMMIT") or "unknown",
        "has_cuda": bool(getattr(compiler, "has_cuda", False)),
        "has_acl": bool(getattr(compiler, "has_acl", False)),
        "use_cuda": bool(getattr(flags, "use_cuda", False)),
        "use_acl": bool(getattr(flags, "use_acl", False)),
        "nvcc_path": getattr(compiler, "nvcc_path", None) or "",
        "tikcc_path": getattr(compiler, "tikcc_path", None) or "",
        "ascend_toolkit_home": environ.get("ASCEND_TOOLKIT_HOME") or "",
        "jittor_home": environ.get("JITTOR_HOME") or "",
        "in_mpi": bool(getattr(jt_module, "in_mpi", False)),
        "rank": getattr(jt_module, "rank", None) or 0,
        "world_size": getattr(jt_module, "world_size", None) or 1,
    }
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
