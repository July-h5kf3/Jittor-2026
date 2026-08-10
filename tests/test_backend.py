import unittest


class Flags:
    def __init__(self):
        self.use_cuda = 0
        self.use_acl = 0


class Compiler:
    def __init__(self, has_cuda=True, has_acl=True):
        self.has_cuda = has_cuda
        self.has_acl = has_acl
        self.nvcc_path = "/opt/cuda/bin/nvcc"
        self.tikcc_path = "/opt/ascend/bin/tikcc"


class FakeJittor:
    def __init__(self, has_cuda=True, has_acl=True):
        self.flags = Flags()
        self.compiler = Compiler(has_cuda=has_cuda, has_acl=has_acl)
        self.rank = 0
        self.world_size = 1
        self.in_mpi = False


class BackendTests(unittest.TestCase):
    def test_acl_enables_acl_and_disables_cuda(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        configure_device("acl", jt)

        self.assertEqual(jt.flags.use_acl, 1)
        self.assertEqual(jt.flags.use_cuda, 0)

    def test_cuda_enables_cuda_and_disables_acl(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        configure_device("cuda", jt)

        self.assertEqual(jt.flags.use_cuda, 1)
        self.assertEqual(jt.flags.use_acl, 0)

    def test_cpu_disables_cuda_and_acl(self):
        from plr3d.backend import configure_device

        jt = FakeJittor()
        jt.flags.use_cuda = 1
        jt.flags.use_acl = 1
        configure_device("cpu", jt)

        self.assertEqual(jt.flags.use_cuda, 0)
        self.assertEqual(jt.flags.use_acl, 0)

    def test_acl_requires_an_available_backend(self):
        from plr3d.backend import BackendError, configure_device

        with self.assertRaisesRegex(BackendError, "ACL backend is unavailable"):
            configure_device("acl", FakeJittor(has_acl=False))

    def test_acl_requires_an_acl_flag(self):
        from plr3d.backend import BackendError, configure_device

        jt = FakeJittor()
        del jt.flags.use_acl

        with self.assertRaisesRegex(BackendError, "ACL backend is unavailable"):
            configure_device("acl", jt)

    def test_status_reports_backend_environment_and_distributed_state(self):
        from plr3d.backend import device_status

        environment = {
            "NKAI_JITTOR_COMMIT": "fake-commit",
            "ASCEND_TOOLKIT_HOME": "/opt/ascend",
            "JITTOR_HOME": "/tmp/jittor-cache",
        }

        status = device_status("acl", FakeJittor(), environment)

        self.assertEqual(status["device"], "acl")
        self.assertEqual(status["jittor_commit"], "fake-commit")
        self.assertEqual(status["ascend_toolkit_home"], "/opt/ascend")
        self.assertEqual(status["jittor_home"], "/tmp/jittor-cache")
        self.assertEqual(status["rank"], 0)
        self.assertEqual(status["world_size"], 1)
        self.assertFalse(status["in_mpi"])
        self.assertTrue(status["has_acl"])
        self.assertEqual(status["tikcc_path"], "/opt/ascend/bin/tikcc")


if __name__ == "__main__":
    unittest.main()
