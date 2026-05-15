"""Tests for `d[key] = value` (index assign) and `a, b = 1, 2` (tuple RHS).

Two LLM-friendly patterns previously rejected by the parser:

  d["b"] = 2       # add or update dict entry
  xs[0] = 99       # set list element
  a, b = 1, 2      # parallel assignment (tuple RHS)
  a, b = b, a      # swap idiom
"""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to


def _run(src: str) -> str:
    py = compile_to(parse(src), "python")
    r = subprocess.run([sys.executable, "-c", py],
                       capture_output=True, text=True, timeout=4)
    return r.stdout.strip() if r.returncode == 0 else f"ERR: {r.stderr.strip()}"


class TestIndexAssign(unittest.TestCase):

    def test_dict_add_entry(self):
        out = _run('d = {"a": 1}\nd["b"] = 2\np d')
        # Python dict repr — order is insertion order in 3.7+.
        self.assertEqual(out, "{'a': 1, 'b': 2}")

    def test_dict_overwrite(self):
        out = _run('d = {"a": 1}\nd["a"] = 99\np d')
        self.assertEqual(out, "{'a': 99}")

    def test_list_element(self):
        out = _run("xs = [10, 20, 30]\nxs[1] = 99\np xs")
        self.assertEqual(out, "[10, 99, 30]")

    def test_value_can_be_expr(self):
        out = _run('d = {"a": 1}\nd["b"] = d["a"] + 1\np d')
        self.assertEqual(out, "{'a': 1, 'b': 2}")

    def test_index_can_be_variable(self):
        out = _run('d = {}\nkey = "name"\nd[key] = "Flow"\np d')
        self.assertEqual(out, "{'name': 'Flow'}")


class TestCompoundIndexAssign(unittest.TestCase):
    """`xs[0] += 5`, `d["k"] *= 2`, etc. — common counter/accumulator idioms."""

    def test_list_plus_eq(self):
        self.assertEqual(_run("xs = [1, 2, 3]\nxs[0] += 5\np xs"),
                         "[6, 2, 3]")

    def test_dict_plus_eq(self):
        self.assertEqual(_run('d = {"a": 1}\nd["a"] += 10\np d'),
                         "{'a': 11}")

    def test_minus_eq(self):
        self.assertEqual(_run('d = {"a": 5}\nd["a"] -= 2\np d'),
                         "{'a': 3}")

    def test_times_eq(self):
        self.assertEqual(_run("xs = [1, 2]\nxs[1] *= 3\np xs"),
                         "[1, 6]")

    def test_div_eq(self):
        self.assertEqual(_run('d = {"x": 10}\nd["x"] /= 2\np d'),
                         "{'x': 5.0}")

    def test_accumulator_pattern(self):
        out = _run(
            'counts = {"hits": 0, "misses": 0}\n'
            'each result in [true, false, true, true]\n'
            '  if result\n'
            '    counts["hits"] += 1\n'
            '  else\n'
            '    counts["misses"] += 1\n'
            "p counts"
        )
        self.assertEqual(out, "{'hits': 3, 'misses': 1}")

    def test_compiles_to_assign(self):
        # Should desugar to `xs[0] = xs[0] + 5`, not `+=` syntax in output.
        py = compile_to(parse("xs = [1]\nxs[0] += 5"), "python")
        # Either form is OK; just check it doesn't error.
        self.assertIn("xs[0]", py)


class TestTupleRHS(unittest.TestCase):

    def test_two_targets_two_values(self):
        out = _run("a, b = 1, 2\np a\np b")
        self.assertEqual(out.splitlines(), ["1", "2"])

    def test_three_targets(self):
        out = _run('a, b, c = 1, "two", true\np f"{a}|{b}|{c}"')
        self.assertEqual(out, "1|two|True")

    def test_swap_idiom(self):
        out = _run("a = 1\nb = 2\na, b = b, a\np f\"{a} {b}\"")
        self.assertEqual(out, "2 1")

    def test_list_rhs_still_works(self):
        # Original behavior: `a, b = [1, 2]` (single RHS that's iterable).
        out = _run("a, b = [10, 20]\np f\"{a} {b}\"")
        self.assertEqual(out, "10 20")


class TestAstShape(unittest.TestCase):

    def test_index_assign_node_used(self):
        from flow.parser import IndexAssignStmt, IndexAccess
        ast = parse('d["b"] = 2')
        stmt = ast.body[0]
        self.assertIsInstance(stmt, IndexAssignStmt)
        self.assertIsInstance(stmt.target, IndexAccess)

    def test_tuple_rhs_becomes_listlit(self):
        from flow.parser import MultiAssignStmt, ListLit
        ast = parse("a, b = 1, 2")
        stmt = ast.body[0]
        self.assertIsInstance(stmt, MultiAssignStmt)
        self.assertIsInstance(stmt.value, ListLit)
        self.assertEqual(len(stmt.value.items), 2)


class TestRoundTrip(unittest.TestCase):

    def test_index_assign_formats_back(self):
        from flow.formatter import format_source
        out = format_source(parse('d["b"] = 2'))
        self.assertIn('d["b"] = 2', out)

    def test_tuple_rhs_formats_as_commas(self):
        from flow.formatter import format_source
        out = format_source(parse("a, b = 1, 2"))
        # Should NOT come back as `[1, 2]`.
        self.assertIn("a, b = 1, 2", out)


class TestCrossTarget(unittest.TestCase):

    def test_js_index_assign(self):
        # Define d so the compiler treats it as a var, not a string literal.
        js = compile_to(parse('d = {}\nd["x"] = 1'), "js")
        self.assertIn('d["x"] = 1', js)

    def test_rust_index_assign_uses_insert(self):
        # Rust has no native dict literal, so build the AST directly to
        # isolate the index-assign emitter.
        from flow.parser import IndexAssignStmt, IndexAccess, Name, StringLit, NumberLit, Program
        prog = Program(body=[
            IndexAssignStmt(
                target=IndexAccess(receiver=Name(["d"]),
                                   index=StringLit("x")),
                value=NumberLit(1),
                line=1,
            )
        ])
        rs = compile_to(prog, "rust")
        self.assertIn(".insert(", rs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
