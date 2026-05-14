"""Parser tests."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.parser import (
    parse, ParseError,
    Program, Call, IfStmt, EachStmt, RepeatStmt, WhenStmt,
    StringLit, NumberLit, BoolLit, Name, FuncCall, BinOp,
    ListLit, DictLit,
)


class TestBasics(unittest.TestCase):

    def test_single_call_no_args(self):
        ast = parse("today -> t")
        self.assertEqual(len(ast.body), 1)
        c = ast.body[0]
        self.assertIsInstance(c, Call)
        self.assertEqual(c.verb, "today")
        self.assertEqual(c.args, [])
        self.assertEqual(c.out, "t")

    def test_call_with_string_arg(self):
        ast = parse('read file="data.csv" -> rows')
        c = ast.body[0]
        self.assertEqual(c.verb, "read")
        self.assertEqual(len(c.args), 1)
        self.assertEqual(c.args[0].name, "file")
        self.assertIsInstance(c.args[0].value, StringLit)
        self.assertEqual(c.args[0].value.value, "data.csv")
        self.assertEqual(c.out, "rows")

    def test_bareword_with_dot(self):
        ast = parse("read file=data.csv -> rows")
        c = ast.body[0]
        v = c.args[0].value
        self.assertIsInstance(v, Name)
        self.assertEqual(v.parts, ["data", "csv"])

    def test_multiple_args(self):
        ast = parse('http_get url="http://x" timeout=5 -> r')
        c = ast.body[0]
        self.assertEqual([a.name for a in c.args], ["url", "timeout"])
        self.assertEqual(c.args[1].value.value, 5)

    def test_print_without_out(self):
        ast = parse("print value=msg")
        c = ast.body[0]
        self.assertIsNone(c.out)

    def test_comment_only(self):
        ast = parse("# nothing here\n# still nothing")
        self.assertEqual(ast.body, [])

    def test_comment_at_end_of_line(self):
        ast = parse("print value=x  # trailing")
        self.assertEqual(len(ast.body), 1)


class TestValues(unittest.TestCase):

    def test_number_int(self):
        ast = parse("wait seconds=5")
        self.assertEqual(ast.body[0].args[0].value.value, 5)

    def test_number_float(self):
        ast = parse("wait seconds=2.5")
        self.assertEqual(ast.body[0].args[0].value.value, 2.5)

    def test_bool_true(self):
        ast = parse("foo flag=true")
        self.assertIsInstance(ast.body[0].args[0].value, BoolLit)
        self.assertTrue(ast.body[0].args[0].value.value)

    def test_bool_false(self):
        ast = parse("foo flag=false")
        self.assertFalse(ast.body[0].args[0].value.value)

    def test_string_with_escapes(self):
        ast = parse(r'print value="line1\nline2"')
        self.assertEqual(ast.body[0].args[0].value.value, "line1\nline2")

    def test_funccall(self):
        ast = parse("foo n=count(items)")
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, FuncCall)
        self.assertEqual(v.name, "count")
        self.assertEqual(len(v.args), 1)

    def test_funccall_no_args(self):
        ast = parse("foo t=now()")
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, FuncCall)
        self.assertEqual(v.args, [])


class TestControl(unittest.TestCase):

    def test_if(self):
        src = "if x > 0\n  print value=hi"
        ast = parse(src)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, IfStmt)
        self.assertIsInstance(stmt.cond, BinOp)
        self.assertEqual(stmt.cond.op, ">")
        self.assertEqual(len(stmt.then), 1)
        self.assertIsNone(stmt.else_)

    def test_if_else(self):
        src = "if x > 0\n  print value=a\nelse\n  print value=b"
        ast = parse(src)
        stmt = ast.body[0]
        self.assertEqual(len(stmt.then), 1)
        self.assertEqual(len(stmt.else_), 1)

    def test_each(self):
        src = "each row in items\n  print value=row"
        ast = parse(src)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, EachStmt)
        self.assertEqual(stmt.var, "row")
        self.assertEqual(stmt.iterable.parts, ["items"])
        self.assertEqual(len(stmt.body), 1)

    def test_repeat(self):
        src = "repeat 3\n  print value=hi"
        ast = parse(src)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, RepeatStmt)
        self.assertEqual(stmt.count.value, 3)

    def test_when(self):
        src = "when start\n  print value=hello"
        ast = parse(src)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, WhenStmt)
        self.assertEqual(stmt.event, "start")

    def test_nested_blocks(self):
        src = (
            "each row in items\n"
            "  if row.age > 18\n"
            "    print value=row.name\n"
            "  else\n"
            "    print value=skip"
        )
        ast = parse(src)
        outer = ast.body[0]
        self.assertIsInstance(outer, EachStmt)
        self.assertEqual(len(outer.body), 1)
        inner = outer.body[0]
        self.assertIsInstance(inner, IfStmt)
        self.assertEqual(len(inner.then), 1)
        self.assertEqual(len(inner.else_), 1)


class TestLiterals(unittest.TestCase):

    def test_empty_list(self):
        ast = parse("foo items=[]")
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, ListLit)
        self.assertEqual(v.items, [])

    def test_list_of_numbers(self):
        ast = parse("foo items=[1, 2, 3]")
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, ListLit)
        self.assertEqual([x.value for x in v.items], [1, 2, 3])

    def test_list_mixed(self):
        ast = parse('foo items=[1, "two", true]')
        v = ast.body[0].args[0].value
        self.assertEqual(len(v.items), 3)
        self.assertIsInstance(v.items[0], NumberLit)
        self.assertIsInstance(v.items[1], StringLit)
        self.assertIsInstance(v.items[2], BoolLit)

    def test_empty_dict(self):
        ast = parse("foo data={}")
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, DictLit)
        self.assertEqual(v.entries, [])

    def test_dict_with_ident_keys(self):
        ast = parse('foo data={a: 1, b: "two"}')
        v = ast.body[0].args[0].value
        self.assertEqual([k for k, _ in v.entries], ["a", "b"])

    def test_dict_with_string_keys(self):
        ast = parse('foo data={"hello world": 1}')
        v = ast.body[0].args[0].value
        self.assertEqual(v.entries[0][0], "hello world")

    def test_nested_list_and_dict(self):
        ast = parse('foo x=[{a: 1}, {a: 2}]')
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, ListLit)
        self.assertIsInstance(v.items[0], DictLit)


class TestErrors(unittest.TestCase):

    def test_missing_equals(self):
        with self.assertRaises(ParseError) as ctx:
            parse("read file data.csv")
        self.assertIn("'='", str(ctx.exception))

    def test_arrow_without_name(self):
        with self.assertRaises(ParseError):
            parse("read file=x ->")

    def test_else_without_if(self):
        with self.assertRaises(ParseError):
            parse("else\n  print value=x")

    def test_each_without_in(self):
        with self.assertRaises(ParseError):
            parse("each row of items\n  print value=row")

    def test_assignment_in_condition(self):
        with self.assertRaises(ParseError) as ctx:
            parse("if x = 1\n  print value=hi")
        self.assertIn("==", str(ctx.exception))

    def test_odd_indent(self):
        with self.assertRaises(ParseError):
            parse("if x > 0\n   print value=x")  # 3 spaces

    def test_tab_indent_rejected(self):
        with self.assertRaises(ParseError):
            parse("if x > 0\n\tprint value=x")


if __name__ == "__main__":
    unittest.main()
