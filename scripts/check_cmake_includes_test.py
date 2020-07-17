#!/usr/bin/env python3
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for check_cmake_includes.py."""

import os
import pathlib
import re
import shutil
import tempfile
from typing import Iterable, Mapping
import unittest
from unittest import mock

import check_cmake_includes


class ConfigureFileParserTest(unittest.TestCase):

  def test_init_positional_args(self):
    path = object()

    parser = check_cmake_includes.ConfigureFileParser(path)

    self.assertIs(parser.path, path)

  def test_init_keyword_args(self):
    path = object()

    parser = check_cmake_includes.ConfigureFileParser(path=path)

    self.assertIs(parser.path, path)

  def test_parse_empty_file_returns_empty_defines(self):
    path = create_temp_file(self)
    parser = check_cmake_includes.ConfigureFileParser(path=path)

    parse_result = parser.parse()

    self.assertEqual(parse_result.defines, frozenset())

  def test_parse_file_with_cmakedefines_returns_those_defines(self):
    configure_file_lines = [
        "#cmakedefine SOME_VAR1",
        "#cmakedefine SOME_VAR2 some_value",
        "#cmakedefine   SOME_VAR3 some_value1 some_value2",
    ]
    path = create_temp_file(self, lines=configure_file_lines)
    parser = check_cmake_includes.ConfigureFileParser(path=path)

    parse_result = parser.parse()

    expected_defines = frozenset(["SOME_VAR1", "SOME_VAR2", "SOME_VAR3"])
    self.assertEqual(parse_result.defines, expected_defines)


class CppFileParserTest(unittest.TestCase):

  def test_init_positional_args(self):
    path = object()

    parser = check_cmake_includes.CppFileParser(path)

    self.assertIs(parser.path, path)

  def test_init_keyword_args(self):
    path = object()

    parser = check_cmake_includes.CppFileParser(path=path)

    self.assertIs(parser.path, path)

  def test_parse_empty_file_returns_empty_result(self):
    path = create_temp_file(self)
    parser = check_cmake_includes.CppFileParser(path=path)

    parse_result = parser.parse()

    self.assert_result(
        parse_result,
        includes=[],
        defines_used={},
    )

  def test_parse(self):
    cpp_file_lines = [
        "#include \"a/b/file_1.h\"", "#include     \"a/b/file_2.h\"",
        "#InCluDe \"file_3.h\"", "#define INTERNAL_DEFINE_1",
        "#define INTERNAL_DEFINE_2 123", "#define   INTERNAL_DEFINE_3 abc",
        "#if VAR1", "#ifdef VAR2", "#ifndef VAR3", "#elif VAR4", "#else",
        "#endif", "#IMadeThisUp VAR_WITH_UNDERSCORES",
        "int main(int argc, char** argv) {", "  return 0;"
        "}"
    ]
    path = create_temp_file(self, lines=cpp_file_lines)
    parser = check_cmake_includes.CppFileParser(path=path)

    parse_result = parser.parse()

    self.assert_result(
        parse_result,
        includes=["a/b/file_1.h", "a/b/file_2.h", "file_3.h"],
        defines_used={
            "VAR1": 7,
            "VAR2": 8,
            "VAR3": 9,
            "VAR4": 10,
            "VAR_WITH_UNDERSCORES": 13,
        },
    )

  def assert_result(
      self,
      parse_result: check_cmake_includes.CppFileParserResult,
      includes: Iterable[str],
      defines_used: Mapping[str, int],
  ) -> None:
    with self.subTest("includes"):
      self.assertEqual(parse_result.includes, frozenset(includes))
    with self.subTest("defines_used"):
      self.assertEqual(parse_result.defines_used, defines_used)


class RequiredIncludesCheckerTest(unittest.TestCase):

  def test_init_positional_args(self):
    defines = object()

    checker = check_cmake_includes.RequiredIncludesChecker(defines)

    self.assertIs(checker.defines, defines)

  def test_init_keyword_args(self):
    defines = object()

    checker = check_cmake_includes.RequiredIncludesChecker(defines=defines)

    self.assertIs(checker.defines, defines)

  def test_check_file_returns_empty_list_if_defines_is_empty(self):
    lines = ["#if HELLO"]
    path = create_temp_file(self, lines=lines)
    checker = check_cmake_includes.RequiredIncludesChecker(defines={})

    missing_includes = checker.check_file(path)

    self.assertEqual(missing_includes, tuple())

  def test_check_file_returns_empty_list_if_no_missing_includes(self):
    lines = [
        "#include \"file_1.h\"",
        "#include \"a/b/file_2.h\"",
        "#define DEFINED_VAR 1",
        "#if VAR_1",
        "#if  VAR_2",
        "#elif UNSPECIFIED_DEFINE",
    ]
    defines = {
        "VAR_1": "file_1.h",
        "VAR_2": "a/b/file_2.h",
        "UNUSED_VAR": "a/b/unused.h",
    }
    path = create_temp_file(self, lines=lines)
    checker = check_cmake_includes.RequiredIncludesChecker(defines=defines)

    missing_includes = checker.check_file(path)

    self.assertEqual(missing_includes, tuple())

  def test_check_file_returns_the_missing_define(self):
    lines = [
        "#if   VAR1",
        "#elif VAR2",
        "#elif VAR2 again",
        "#ifndef UNSPECIFIED_DEFINE",
    ]
    defines = {
        "VAR1": "file1.h",
        "VAR2": "file2.h",
    }
    path = create_temp_file(self, lines=lines)
    checker = check_cmake_includes.RequiredIncludesChecker(defines=defines)

    missing_includes = checker.check_file(path)

    expected_missing_includes = (
        check_cmake_includes.MissingInclude(
            define="VAR1", include="file1.h", line_number=1),
        check_cmake_includes.MissingInclude(
            define="VAR2", include="file2.h", line_number=2),
    )
    self.assertCountEqual(missing_includes, expected_missing_includes)


class RunTest(unittest.TestCase):

  def test_no_errors(self):
    cmake_configure_file_lines = [
        "blah blah",
        "#cmakedefine VAR1",
        "#cmakedefine VAR2 1",
        "blah blah",
    ]
    cmake_configure_file = create_temp_file(self, cmake_configure_file_lines)
    source_file_lines = [
        "blah blah",
        "#define ABC",
        "#include \"a/b/c.h\"",
        "#ifdef VAR1",
        "# if  VAR2 zzz",
        "#elif XYZ",
    ]
    source_file = create_temp_file(self, source_file_lines)
    output_lines = []

    num_errors = check_cmake_includes.run(
        cmake_configure_files={cmake_configure_file: "a/b/c.h"},
        source_files=[source_file],
        print_func=output_lines.append,
    )

    with self.subTest("num_errors"):
      self.assertEqual(num_errors, 0)
    with self.subTest("output_lines"):
      self.assertGreater(len(output_lines), 0)
      self.assert_contains_word(output_lines[0], "scanned 1", ignore_case=True)
      self.assert_contains_word(output_lines[0], "0 error(s)", ignore_case=True)
      self.assertEqual(len(output_lines), 1)

  def test_with_errors(self):
    cmake_configure_file1 = create_temp_file(self, [
        "#cmakedefine VAR1",
    ])
    cmake_configure_file2 = create_temp_file(self, [
        "#cmakedefine VAR2",
    ])
    valid_source_file1 = create_temp_file(self, [
        "#include \"config1.h\"",
        "#include \"config2.h\"",
        "#if VAR1",
        "#if VAR2",
    ])
    valid_source_file2 = create_temp_file(self, [
        "#if SOME_OTHER_VAR1",
        "#if SOME_OTHER_VAR2",
    ])
    missing_config1_source_file = create_temp_file(self, [
        "#include \"config2.h\"",
        "#if VAR1",
        "#if VAR2",
    ])
    missing_config2_source_file = create_temp_file(self, [
        "#include \"config1.h\"",
        "#if VAR1",
        "#if VAR2",
    ])
    missing_config1and2_source_file = create_temp_file(self, [
        "#if VAR1",
        "#if VAR2",
    ])
    output_lines = []

    num_errors = check_cmake_includes.run(
        cmake_configure_files={
            cmake_configure_file1: "config1.h",
            cmake_configure_file2: "config2.h",
        },
        source_files=[
            valid_source_file1,
            valid_source_file2,
            missing_config1_source_file,
            missing_config2_source_file,
            missing_config1and2_source_file,
        ],
        print_func=output_lines.append,
    )

    with self.subTest("num_errors"):
      self.assertEqual(num_errors, 4)
    with self.subTest("output_lines[0]"):
      self.assertGreater(len(output_lines), 0)
      self.assert_missing_include_line(
          line=output_lines[0],
          source_file=missing_config1_source_file,
          var_name="VAR1",
          missing_include="config1.h",
          line_number=2,
      )
    with self.subTest("output_lines[1]"):
      self.assertGreater(len(output_lines), 1)
      self.assert_missing_include_line(
          line=output_lines[1],
          source_file=missing_config2_source_file,
          var_name="VAR2",
          missing_include="config2.h",
          line_number=3,
      )
    with self.subTest("output_lines[2]"):
      self.assertGreater(len(output_lines), 2)
      self.assert_missing_include_line(
          line=output_lines[2],
          source_file=missing_config1and2_source_file,
          var_name="VAR1",
          missing_include="config1.h",
          line_number=1,
      )
    with self.subTest("output_lines[3]"):
      self.assertGreater(len(output_lines), 3)
      self.assert_missing_include_line(
          line=output_lines[3],
          source_file=missing_config1and2_source_file,
          var_name="VAR2",
          missing_include="config2.h",
          line_number=2,
      )
    with self.subTest("output_lines[4]"):
      self.assertGreater(len(output_lines), 4)
      self.assert_contains_word(output_lines[4], "scanned 5", ignore_case=True)
      self.assert_contains_word(output_lines[4], "4 error(s)", ignore_case=True)
    with self.subTest("len(output_lines)"):
      self.assertEqual(len(output_lines), 5)

  def assert_missing_include_line(
      self,
      line: str,
      source_file: pathlib.Path,
      var_name: str,
      missing_include: str,
      line_number: int,
  ) -> None:
    self.assert_contains_word(line, f"{source_file}:{line_number}")
    self.assert_contains_word(line, var_name)
    self.assert_contains_word(line, missing_include)

  def assert_contains_word(
      self,
      text: str,
      word: str,
      ignore_case: bool = False,
  ) -> None:
    pattern = (("(?i)" if ignore_case else "") + r"(\W|^)" + re.escape(word) +
               r"(\W|$)")
    self.assertRegex(text, pattern)


class ArgumentParserTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.parser = check_cmake_includes.ArgumentParser()
    mock.patch.object(
        self.parser.parser,
        "_print_message",
        spec_set=True,
        autospec=True,
    ).start()
    mock.patch.object(
        self.parser.parser,
        "exit",
        spec_set=True,
        autospec=True,
        side_effect=self.mock_argument_parser_exit,
    ).start()

  def test_no_args_should_fail(self):
    with self.assertRaises(self.MockArgparseExitError):
      self.parser.parse_args([])

  def test_source_files_not_specified_should_fail(self):
    with self.assertRaises(self.MockArgparseExitError) as assert_context:
      self.parser.parse_args([
          "--cmake_configure_file=cmake_configure_file.txt",
          "--required_include=required_include.h",
      ])

    exception = assert_context.exception
    self.assertEqual(exception.status, 2)
    self.assertIn("source_files", exception.message)

  def test_cmake_configure_file_not_specified_should_fail(self):
    with self.assertRaises(self.MockArgparseExitError) as assert_context:
      self.parser.parse_args([
          "--required_include=required_include.h",
          "main.cc",
      ])

    exception = assert_context.exception
    self.assertEqual(exception.status, 2)
    self.assertIn("--cmake_configure_file", exception.message)

  def test_required_include_not_specified_should_fail(self):
    with self.assertRaises(self.MockArgparseExitError) as assert_context:
      self.parser.parse_args([
          "--cmake_configure_file=cmake_configure_file.txt",
          "main.cc",
      ])

    exception = assert_context.exception
    self.assertEqual(exception.status, 2)
    self.assertIn("--required_include", exception.message)

  def test_non_existent_source_files_should_be_treated_like_a_file(self):
    temp_dir = create_temp_dir(self)
    non_existent_file = temp_dir / "IDoNotExist.cc"

    parse_result = self.parser.parse_args([
        "--cmake_configure_file=cmake_configure_file.txt",
        "--required_include=required_include.h",
        f"{non_existent_file}",
    ])

    self.assertEqual(parse_result.source_files, [non_existent_file])

  def test_source_files_that_are_files_should_be_returned(self):
    source_file1 = create_temp_file(self)
    source_file2 = create_temp_file(self)
    source_file3 = create_temp_file(self)

    parse_result = self.parser.parse_args([
        "--cmake_configure_file=cmake_configure_file.txt",
        "--required_include=required_include.h",
        f"{source_file1}",
        f"{source_file2}",
        f"{source_file3}",
    ])

    self.assertEqual(parse_result.source_files, [
        source_file1,
        source_file2,
        source_file3,
    ])

  def test_source_files_that_are_directories_should_be_recursed(self):
    source_dir = create_temp_dir(self)
    source_file1 = source_dir / "src1.cc"
    source_file1.touch()
    source_file2 = source_dir / "src2.cc"
    source_file2.touch()
    subdir1 = source_dir / "subdir1"
    subdir1.mkdir()
    source_file3 = subdir1 / "src3.cc"
    source_file3.touch()
    source_file4 = subdir1 / "src4.cc"
    source_file4.touch()
    subdir2 = source_dir / "subdir2"
    subdir2.mkdir()
    source_file5 = subdir2 / "src5.cc"
    source_file5.touch()
    source_file6 = subdir2 / "src6.cc"
    source_file6.touch()

    parse_result = self.parser.parse_args([
        "--cmake_configure_file=cmake_configure_file.txt",
        "--required_include=required_include.h",
        f"{source_dir}",
    ])

    self.assertCountEqual(parse_result.source_files, [
        source_file1,
        source_file2,
        source_file3,
        source_file4,
        source_file5,
        source_file6,
    ])

  def test_filename_includes_are_respected(self):
    source_dir = create_temp_dir(self)
    source_file1 = source_dir / "src.cc"
    source_file1.touch()
    source_file2 = source_dir / "src.h"
    source_file2.touch()
    source_file3 = source_dir / "src.txt"
    source_file3.touch()
    source_file4 = source_dir / "src.h.in"
    source_file4.touch()
    source_file5 = source_dir / "wwXzz"
    source_file5.touch()
    source_file6 = source_dir / "wwYzz"
    source_file6.touch()

    parse_result = self.parser.parse_args([
        "--cmake_configure_file=cmake_configure_file.txt",
        "--required_include=required_include.h",
        "--filename_include=*.cc",
        "--filename_include=*.h",
        "--filename_include=*[XY]*",
        f"{source_dir}",
    ])

    self.assertCountEqual(parse_result.source_files, [
        source_file1,
        source_file2,
        source_file5,
        source_file6,
    ])

  def test_fewer_cmake_configure_files_than_required_includes(self):
    with self.assertRaises(self.MockArgparseExitError) as assert_context:
      self.parser.parse_args([
          "--cmake_configure_file",
          "cmake_configure_file1.txt",
          "--required_include",
          "required_include1.h",
          "--required_include",
          "required_include2.h",
          "main.cc",
      ])

    exception = assert_context.exception
    self.assertEqual(exception.status, 2)
    self.assertIn("--required_include", exception.message)
    self.assertIn("--cmake_configure_file", exception.message)
    self.assertIn(" 2 ", exception.message)
    self.assertIn(" 1 ", exception.message)

  def test_more_cmake_configure_files_than_required_includes(self):
    with self.assertRaises(self.MockArgparseExitError) as assert_context:
      self.parser.parse_args([
          "--cmake_configure_file",
          "cmake_configure_file1.txt",
          "--cmake_configure_file",
          "cmake_configure_file2.txt",
          "--cmake_configure_file",
          "cmake_configure_file3.txt",
          "--required_include",
          "required_include1.h",
          "main.cc",
      ])

    exception = assert_context.exception
    self.assertEqual(exception.status, 2)
    self.assertIn("--required_include", exception.message)
    self.assertIn("--cmake_configure_file", exception.message)
    self.assertIn(" 3 ", exception.message)
    self.assertIn(" 1 ", exception.message)

  def test_cmake_configure_files_and_required_includes(self):
    parse_result = self.parser.parse_args([
        "--cmake_configure_file",
        "cmake_configure_file1.txt",
        "--cmake_configure_file",
        "cmake_configure_file2.txt",
        "--required_include",
        "required_include1.h",
        "--required_include",
        "required_include2.h",
        "main.cc",
    ])

    self.assertEqual(
        parse_result.cmake_configure_files, {
            pathlib.Path("cmake_configure_file1.txt"): "required_include1.h",
            pathlib.Path("cmake_configure_file2.txt"): "required_include2.h",
        })

  class MockArgparseExitError(Exception):
    """Exception raised by the mocked ArgumentParser methods."""

    def __init__(self, status, message):
      super().__init__(f"status={status} message={message}")
      self.status = status
      self.message = message

  def mock_argument_parser_exit(self, status=0, message=None):
    raise self.MockArgparseExitError(status, message)


class MainTest(unittest.TestCase):

  def test_no_errors(self):
    root_dir = create_temp_dir(self)
    src_dir = root_dir / "src"
    src_dir.mkdir()
    include_dir = root_dir / "include"
    include_dir.mkdir()
    config_h_in_file = write_lines_to_file(
        include_dir / "config.h.in",
        "#cmakedefine VAR1",
        "#cmakedefine VAR2",
    )
    write_lines_to_file(
        src_dir / "file1.cc",
        "#include \"config.h\"",
        "#if VAR1",
    )
    write_lines_to_file(
        src_dir / "file2.cc",
        "#include \"config.h\"",
        "#if VAR2",
    )
    write_lines_to_file(
        src_dir / "ignore_me.txt",
        "// The line below would be an error if scanned.",
        "#if VAR2",
    )
    args = [
        f"--cmake_configure_file={config_h_in_file}",
        "--required_include=config.h",
        "--filename_include=*.cc",
        f"{root_dir}",
    ]
    print_func = mock.create_autospec(print, spec_set=True)

    exit_code = check_cmake_includes.main(args=args, print_func=print_func)

    self.assertEqual(exit_code, 0)

  def test_with_errors(self):
    root_dir = create_temp_dir(self)
    config_h_in_file = write_lines_to_file(
        root_dir / "config.h.in",
        "#cmakedefine VAR1",
        "#cmakedefine VAR2",
    )
    write_lines_to_file(root_dir / "file1.cc", "#if VAR1")
    write_lines_to_file(root_dir / "file2.cc", "#if VAR2")
    args = [
        f"--cmake_configure_file={config_h_in_file}",
        "--required_include=config.h",
        "--filename_include=*.cc",
        f"{root_dir}",
    ]
    print_func = mock.create_autospec(print, spec_set=True)

    exit_code = check_cmake_includes.main(args=args, print_func=print_func)

    self.assertEqual(exit_code, 1)


def create_temp_file(
    test_case: unittest.TestCase,
    lines: Iterable[str] = tuple(),
) -> pathlib.Path:
  (handle, path_str) = tempfile.mkstemp()
  os.close(handle)
  test_case.addCleanup(os.remove, path_str)
  lines = lines if lines is not None else []
  return write_lines_to_file(pathlib.Path(path_str), *lines)


def write_lines_to_file(path: pathlib.Path, *lines: str) -> pathlib.Path:
  with path.open("wt", encoding="utf8") as f:
    for line in lines:
      print(line, file=f)
  return path


def create_temp_dir(test_case: unittest.TestCase) -> pathlib.Path:
  path_str = tempfile.mkdtemp()
  test_case.addCleanup(shutil.rmtree, path_str)
  return pathlib.Path(path_str)


if __name__ == "__main__":
  unittest.main()
