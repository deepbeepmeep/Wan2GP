"""Offline tests for the launcher's platform-independent logic.

These run anywhere: they never import pywebview and never touch a GPU. The
Windows-only paths (installer flow, WebView2 window) are covered by the
smoke-test step of the build workflow instead.
"""

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from launcher import config, environment, process, server  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_defaults_are_complete(self):
        cfg = config.load_config()
        for key in ("app_name", "app_version", "repo_url", "python", "server", "window"):
            self.assertIn(key, cfg)
        self.assertEqual(cfg["python"]["series"], "3.11")

    def test_bundled_json_overrides_defaults(self):
        cfg = config.load_config()
        # launcher_config.json ships next to the module and must win.
        self.assertTrue(cfg["repo_url"].endswith(".git"))
        self.assertIn("python.org", cfg["python"]["installer_url"])

    def test_install_root_is_the_checkout_from_source(self):
        root = config.install_root()
        self.assertTrue(os.path.isfile(os.path.join(root, "wgp.py")))
        self.assertTrue(os.path.isfile(os.path.join(root, "setup.py")))

    def test_setup_page_is_where_the_ui_looks_for_it(self):
        self.assertTrue(os.path.isfile(os.path.join(config.bundle_dir(), "web", "setup.html")))


class ExtraArgsTests(unittest.TestCase):
    def _root_with_args(self, contents):
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "scripts"))
        with open(os.path.join(root, "scripts", "args.txt"), "w", encoding="utf-8") as handle:
            handle.write(contents)
        return root

    def test_missing_file_yields_no_args(self):
        self.assertEqual(environment.read_extra_args(tempfile.mkdtemp()), [])

    def test_flags_are_split_and_comments_ignored(self):
        root = self._root_with_args("# comment\n--advanced --profile 3\n\n--fp16\n")
        self.assertEqual(environment.read_extra_args(root), ["--advanced", "--profile", "3", "--fp16"])

    def test_browser_and_server_flags_are_stripped(self):
        root = self._root_with_args("--advanced --open-browser --server-port 9000 --share --listen --fp16\n")
        self.assertEqual(environment.read_extra_args(root), ["--advanced", "--fp16"])

    def test_equals_form_of_server_options_is_stripped(self):
        root = self._root_with_args("--server-name=0.0.0.0 --steps 20\n")
        self.assertEqual(environment.read_extra_args(root), ["--steps", "20"])


class EnvPythonTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def _make(self, *parts):
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    def test_venv_uses_the_scripts_layout(self):
        expected = self._make("env_venv", "Scripts", "python.exe")
        self.assertEqual(environment.env_python(self.root, "venv", "./env_venv", "sys"), expected)

    def test_conda_interpreter_sits_at_the_env_root(self):
        expected = self._make("env_conda", "python.exe")
        self.assertEqual(environment.env_python(self.root, "conda", "./env_conda", "sys"), expected)

    def test_env_type_none_falls_back_to_the_system_interpreter(self):
        self.assertEqual(environment.env_python(self.root, "none", "", "/usr/bin/python3"), "/usr/bin/python3")

    def test_missing_interpreter_is_an_actionable_error(self):
        with self.assertRaises(environment.EnvironmentError_) as caught:
            environment.env_python(self.root, "venv", "./gone", "sys")
        self.assertIn("Repair", str(caught.exception))


class SetupBridgeTests(unittest.TestCase):
    """get_env_info must parse exactly what setup.py prints, as run.bat does."""

    def _fake_checkout(self, setup_body):
        root = tempfile.mkdtemp()
        with open(os.path.join(root, "setup.py"), "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent(setup_body))
        open(os.path.join(root, "wgp.py"), "w").close()
        return root

    def test_parses_the_env_info_line(self):
        root = self._fake_checkout(
            """
            import sys
            print("[*] chatter that must be ignored")
            print("ENV_INFO|venv|./env_venv")
            sys.exit(0)
            """
        )
        self.assertEqual(environment.get_env_info(root, sys.executable), ("venv", "./env_venv"))

    def test_non_zero_exit_means_no_environment(self):
        root = self._fake_checkout(
            """
            import sys
            sys.exit(1)
            """
        )
        self.assertIsNone(environment.get_env_info(root, sys.executable))

    def test_is_wangp_checkout_requires_both_scripts(self):
        root = self._fake_checkout("pass")
        self.assertTrue(environment.is_wangp_checkout(root))
        os.remove(os.path.join(root, "wgp.py"))
        self.assertFalse(environment.is_wangp_checkout(root))


class ProcessTests(unittest.TestCase):
    def test_stream_forwards_every_line_and_returns_the_exit_code(self):
        script = "import sys; [print('line %d' % i) for i in range(3)]; sys.exit(7)"
        lines = []
        code = process.stream([sys.executable, "-c", script], on_line=lines.append)
        self.assertEqual(code, 7)
        self.assertEqual(lines, ["line 0", "line 1", "line 2"])

    def test_children_receive_newlines_so_input_prompts_do_not_crash(self):
        # setup.py calls input() when several environments exist; without the
        # pre-fed stdin that would raise EOFError and abort the update.
        script = "answer = input('pick: '); print('got[%s]' % answer.strip())"
        lines = []
        code = process.stream([sys.executable, "-c", script], on_line=lines.append)
        self.assertEqual(code, 0)
        self.assertIn("got[]", "".join(lines))

    def test_cancellation_stops_the_stream(self):
        script = "import sys, time\nfor i in range(1000):\n    print(i); sys.stdout.flush(); time.sleep(0.01)"
        lines = []
        code = process.stream(
            [sys.executable, "-u", "-c", script],
            on_line=lines.append,
            cancelled=lambda: len(lines) >= 3,
        )
        self.assertEqual(code, -1)
        self.assertLess(len(lines), 50)

    def test_run_quiet_captures_output(self):
        code, out = process.run_quiet([sys.executable, "-c", "print('hello')"])
        self.assertEqual(code, 0)
        self.assertEqual(out, "hello")

    def test_child_env_drops_pyinstaller_leakage(self):
        os.environ["PYTHONHOME"] = "/should/not/leak"
        try:
            env = process.child_env()
            self.assertNotIn("PYTHONHOME", env)
            self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        finally:
            os.environ.pop("PYTHONHOME", None)


class PortTests(unittest.TestCase):
    def test_preferred_port_is_used_when_free(self):
        free = server.pick_port("127.0.0.1", 0)
        self.assertGreater(free, 0)

    def test_busy_preferred_port_falls_back(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            busy = taken.getsockname()[1]
            chosen = server.pick_port("127.0.0.1", busy)
            self.assertNotEqual(chosen, busy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
