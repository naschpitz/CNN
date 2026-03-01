#!/usr/bin/env python3
"""
Ensure blank lines before and after if/else/try/catch blocks in C++ files.

Rules:
  - Insert a blank line BEFORE an if/try line, unless:
    - It is the first non-blank line of a block (i.e., preceded by '{' only)
    - It is already preceded by a blank line
    - It is preceded by a comment line
    - It is preceded by another closing brace '}'
  - Insert a blank line AFTER the closing '}' of an if/else if/else/try/catch
    chain, unless:
    - The next non-blank line is '}' (end of enclosing block)
    - The next non-blank line is 'else' or 'catch' (continuation of the chain)
    - There is already a blank line after it
    - The next line is a comment
"""

import re
import sys

# Patterns that START a block chain
BLOCK_START_RE = re.compile(r'^\s*(if\s*\(|try\s*\{|try\s*$)')

# Patterns that CONTINUE a chain (must not insert blank line before these)
CHAIN_CONTINUE_RE = re.compile(r'^\s*(else\s|else\s*\{|catch\s*\()')

# A line that is just a closing brace (possibly with trailing comment), but NOT
# followed by else/catch on the same line (e.g. "} else {" is a chain, not a close)
CLOSING_BRACE_RE = re.compile(r'^\s*\}')

# A closing brace that continues a chain on the same line: "} else {", "} catch (...) {"
BRACE_CHAIN_RE = re.compile(r'^\s*\}\s*(else|catch)\b')

# A comment-only line
COMMENT_RE = re.compile(r'^\s*//')

# An opening brace only line
OPENING_BRACE_RE = re.compile(r'^\s*\{\s*$')

# A line that ends with an opening brace (e.g. function signature)
ENDS_WITH_BRACE_RE = re.compile(r'\{\s*$')

# Preprocessor directive
PREPROCESSOR_RE = re.compile(r'^\s*#')

# Line continuation (macro body)
LINE_CONTINUATION_RE = re.compile(r'\\\s*$')


def is_blank(line):
  return line.strip() == ''


def in_macro_body(lines, i):
  """Check if line i is inside a multi-line macro (previous line ends with \\)."""
  if i > 0 and LINE_CONTINUATION_RE.search(lines[i - 1]):
    return True

  return False


def process_file(filepath):
  with open(filepath, 'r') as f:
    lines = f.readlines()

  result = []
  i = 0
  n = len(lines)
  format_off = False

  while i < n:
    line = lines[i]
    stripped = line.strip()

    # Track clang-format off/on regions
    if stripped == '// clang-format off':
      format_off = True

    if stripped == '// clang-format on':
      format_off = False

    # Skip formatting-off regions and preprocessor/macro lines
    if format_off or PREPROCESSOR_RE.match(line) or in_macro_body(lines, i):
      result.append(line)
      i += 1
      continue

    # --- Insert blank line BEFORE if/try ---
    if BLOCK_START_RE.match(line):
      if result:
        prev = result[-1]
        prev_stripped = prev.strip()

        needs_blank = (
          not is_blank(prev)
          and not OPENING_BRACE_RE.match(prev)
          and not ENDS_WITH_BRACE_RE.search(prev)
          and not COMMENT_RE.match(prev)
          and not CLOSING_BRACE_RE.match(prev)
        )

        if needs_blank:
          result.append('\n')

    result.append(line)

    # --- Insert blank line AFTER closing '}' of a block chain ---
    # We detect: current line is '}' and it ends an if/else/try/catch chain.
    # Heuristic: if the NEXT non-blank line is NOT else/catch/'}', insert blank.
    # But we only do this if the previous block was an if/else/try/catch.
    # Since tracking brace depth is complex, we use a simpler approach:
    # if current line is '}' and the next non-blank line is a statement (not
    # else/catch/'}'/blank/comment), AND the line before the matching '{' was
    # if/else/try/catch, insert a blank line.
    #
    # Simplified: if current line closes a brace and next line is code (not
    # else/catch/}/blank/comment), check if we should insert a blank.
    # We track this by looking at what follows.

    if CLOSING_BRACE_RE.match(line) and not CHAIN_CONTINUE_RE.match(line) and not BRACE_CHAIN_RE.match(line):
      # Look ahead to find next non-blank line
      j = i + 1

      while j < n and is_blank(lines[j]):
        j += 1

      if j < n:
        next_line = lines[j]

        # Don't insert if next is else/catch (chain continues)
        if CHAIN_CONTINUE_RE.match(next_line):
          i += 1
          continue

        # Don't insert if next is another closing brace
        if CLOSING_BRACE_RE.match(next_line):
          i += 1
          continue

        # Don't insert if next is a comment
        if COMMENT_RE.match(next_line):
          i += 1
          continue

        # Don't insert if there's already a blank line between
        if i + 1 < n and is_blank(lines[i + 1]):
          i += 1
          continue

        # Insert blank line after the closing brace
        result.append('\n')

    i += 1

  # Write back
  with open(filepath, 'w') as f:
    f.writelines(result)


def main():
  changed = False

  for filepath in sys.argv[1:]:
    with open(filepath, 'r') as f:
      original = f.read()

    process_file(filepath)

    with open(filepath, 'r') as f:
      modified = f.read()

    if original != modified:
      changed = True

  sys.exit(1 if changed else 0)


if __name__ == '__main__':
  main()

