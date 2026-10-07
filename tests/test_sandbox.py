"""Tests for sandbox.py: static safety checks and subprocess execution of generated Python."""

import pytest

from viki_slm_125m.sft import sandbox


# ---------- static safety ----------

@pytest.mark.parametrize("code", [
    "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'a':[1,2]})\nprint(df.a.sum())",
    "from sklearn.linear_model import LinearRegression\nimport math\nprint(math.sqrt(4))",
    "import matplotlib.pyplot as plt\nplt.plot([1,2,3])\nplt.savefig('p.png')",
    "x = [i*i for i in range(5)]\nprint(sum(x))",
])
def test_safe_code_is_accepted(code):
    ok, reason = sandbox.check_code_safe(code)
    assert ok, reason


@pytest.mark.parametrize("code,needle", [
    ("import os\nos.system('echo hi')", "import"),
    ("import subprocess", "import"),
    ("from socket import socket", "import"),
    ("exec('print(1)')", "exec"),
    ("eval('1+1')", "eval"),
    ("open('/etc/passwd').read()", "open"),
    ("__import__('os')", "__import__"),
    ("x = ().__class__.__bases__", "dunder"),
    ("getattr(object, 'x')", "getattr"),
    ("import pandas as pd\npd.read_csv('http://evil.example/data.csv')", "url"),
    ("import pandas as pd\npd.read_csv('/etc/hosts')", "path"),
    ("import pandas as pd\npd.read_csv('../secret.csv')", "path"),
    ("def f(:\n  pass", "syntax"),
])
def test_unsafe_code_is_rejected_with_reason(code, needle):
    ok, reason = sandbox.check_code_safe(code)
    assert not ok and needle in reason.lower()


# ---------- execution ----------

def test_run_python_captures_stdout():
    r = sandbox.run_python("print(2 + 3)")
    assert r.ok and r.stdout.strip() == "5" and not r.timed_out


def test_run_python_reports_errors_with_traceback_tail():
    r = sandbox.run_python("x = 1\nprint(undefined_name)")
    assert not r.ok and "NameError" in r.error


def test_run_python_times_out():
    r = sandbox.run_python("while True:\n    pass", timeout_s=1.0)
    assert r.timed_out and not r.ok and "timeout" in r.error.lower()


def test_run_python_provides_files_in_working_directory():
    r = sandbox.run_python("import pandas as pd\nprint(pd.read_csv('d.csv').a.sum())",
                           files={"d.csv": "a\n1\n2\n3\n"})
    assert r.ok and r.stdout.strip() == "6"


def test_run_python_rejects_unsafe_code_without_running(tmp_path):
    marker = tmp_path / "touched.txt"
    r = sandbox.run_python(f"import os\nopen(r'{marker}', 'w').write('x')")
    assert not r.ok and not marker.exists() and "rejected" in r.error.lower()


def test_run_python_truncates_long_output():
    r = sandbox.run_python("print('x' * 20000)", max_output=500)
    assert r.ok and len(r.stdout) <= 600 and "truncated" in r.stdout
