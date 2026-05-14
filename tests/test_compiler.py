"""Compiler tests — verify the produced Python actually runs."""
import sys
import os
import io
import subprocess
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, CompileError


def _run_python(src: str) -> str:
    """Run a Python source string in a subprocess; return stdout."""
    result = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Generated Python failed:\n----SRC----\n{src}\n----STDERR----\n{result.stderr}"
        )
    return result.stdout


class TestCompileBasics(unittest.TestCase):

    def test_print_string(self):
        py = compile_to(parse('print value="hello"'), "python")
        self.assertIn('print("hello")', py)
        self.assertEqual(_run_python(py).strip(), "hello")

    def test_print_variable(self):
        src = 'today -> d\nprint value=d'
        py = compile_to(parse(src), "python")
        # smoke: should not raise; today() is a real builtin verb
        self.assertIn("print(d)", py)

    def test_math_add(self):
        src = "add a=2 b=3 -> s\nprint value=s"
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_python(py).strip(), "5")

    def test_each_loop(self):
        src = textwrap.dedent("""\
            each n in [1, 2, 3]
              print value=n
        """)
        # NOTE: list literal not yet supported by parser; use load or build differently
        # Skip — list literal support is a future extension.
        # Instead, test each over a string/funccall.
        src = textwrap.dedent("""\
            split text="a,b,c" sep="," -> parts
            each p in parts
              print value=p
        """)
        py = compile_to(parse(src), "python")
        out = _run_python(py)
        self.assertEqual(out.strip().split("\n"), ["a", "b", "c"])

    def test_if_else(self):
        src = textwrap.dedent("""\
            add a=10 b=5 -> s
            if s > 12
              print value="big"
            else
              print value="small"
        """)
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_python(py).strip(), "big")

    def test_repeat(self):
        src = textwrap.dedent("""\
            repeat 3
              print value="hi"
        """)
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_python(py).strip().split("\n"), ["hi", "hi", "hi"])

    def test_when_start_wraps_in_main(self):
        src = textwrap.dedent("""\
            when start
              print value="ok"
        """)
        py = compile_to(parse(src), "python")
        self.assertIn("def main():", py)
        self.assertIn('if __name__ == "__main__":', py)
        self.assertEqual(_run_python(py).strip(), "ok")

    def test_filter_with_raw_where(self):
        src = textwrap.dedent("""\
            split text="1,2,3,4,5" sep="," -> nums
            map from=nums to="int(x)" -> ns
            filter from=ns where="x > 2" -> big
            count of=big -> n
            print value=n
        """)
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_python(py).strip(), "3")


class TestLiteralsCompile(unittest.TestCase):

    def test_list_python(self):
        py = compile_to(parse('sum of=[1, 2, 3] -> s\nprint value=s'), "python")
        self.assertEqual(_run_python(py).strip(), "6")

    def test_dict_python_format(self):
        src = (
            'format template="hi {name}, age {age}" data={name: "alice", age: 30} -> msg\n'
            'print value=msg'
        )
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_python(py).strip(), "hi alice, age 30")

    def test_dict_js_renders(self):
        js = compile_to(parse('save value={name: "x"} file="/tmp/out.json"'), "js")
        self.assertIn('{name: "x"}', js)


class TestCompileErrors(unittest.TestCase):

    def test_unknown_verb(self):
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse("doesnotexist x=1"), "python")
        self.assertIn("unknown verb", str(ctx.exception))

    def test_bad_arg(self):
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse('read nonsense="x"'), "python")
        self.assertIn("doesn't accept arg", str(ctx.exception))

    def test_out_on_non_returning_verb(self):
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse('print value=x -> dummy'), "python")
        self.assertIn("does not return", str(ctx.exception))

    def test_unknown_target(self):
        with self.assertRaises(CompileError):
            compile_to(parse("print value=1"), "ruby")


class TestJsCompile(unittest.TestCase):
    """JS output is smoke-tested (no node available necessarily)."""

    def test_js_print(self):
        js = compile_to(parse('print value="hi"'), "js")
        self.assertIn('console.log("hi");', js)

    def test_js_if(self):
        js = compile_to(parse('add a=1 b=2 -> n\nif n > 0\n  print value="y"'), "js")
        self.assertIn("if ((n > 0)) {", js)


class TestExtraTargets(unittest.TestCase):
    """Smoke tests for go/rust/bash — syntactic shape only."""

    SAMPLE = (
        'add a=3 b=4 -> s\n'
        'if s > 5\n'
        '  print value="big"\n'
        'else\n'
        '  print value="small"'
    )

    def test_go_shape(self):
        go = compile_to(parse(self.SAMPLE), "go")
        self.assertIn("var s = (3) + (4)", go)
        self.assertIn("if (s > 5) {", go)
        self.assertIn("} else {", go)            # idiomatic else
        self.assertIn('fmt.Println("big")', go)

    def test_rust_shape(self):
        rs = compile_to(parse(self.SAMPLE), "rust")
        self.assertIn("let s = (3) + (4);", rs)
        self.assertIn("if (s > 5) {", rs)
        self.assertIn("} else {", rs)
        self.assertIn('println!', rs)

    def test_bash_shape(self):
        sh = compile_to(parse(self.SAMPLE), "bash")
        self.assertIn("s=$(( 3 + 4 ))", sh)
        self.assertIn("if ((", sh)
        self.assertIn("then", sh)
        self.assertIn("fi", sh)
        self.assertIn('echo "big"', sh)

    def test_bash_uses_dollar_for_vars(self):
        sh = compile_to(parse("add a=1 b=2 -> n\nprint value=n"), "bash")
        self.assertIn("echo $n", sh)

    def test_when_start_wraps(self):
        src = 'when start\n  print value="ok"'
        self.assertIn("func main()",  compile_to(parse(src), "go"))
        self.assertIn("fn main()",    compile_to(parse(src), "rust"))
        self.assertIn("main() {",     compile_to(parse(src), "bash"))

    def test_each_loop_go(self):
        src = 'split text="a,b,c" sep="," -> parts\neach p in parts\n  print value=p'
        go = compile_to(parse(src), "go")
        self.assertIn("for _, p := range parts", go)

    def test_each_loop_rust(self):
        src = 'split text="a,b,c" sep="," -> parts\neach p in parts\n  print value=p'
        rs = compile_to(parse(src), "rust")
        self.assertIn("for p in parts", rs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
