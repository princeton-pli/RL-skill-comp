### Written with the help of OpenAI o3 model ###
import collections
import json
import os
import re

from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from tqdm import tqdm
from typing import List

# ──────────────────────────────────────────────────────────────
# 1.  Basic arithmetic operations (exact, with zero-check on /)
# ──────────────────────────────────────────────────────────────
OPS = {
    '+': (lambda a, b: a + b),
    '-': (lambda a, b: a - b),
    '*': (lambda a, b: a * b),
    '/': (lambda a, b: a / b if b != 0 else None),
}

def get_size(pattern: str) -> int:
    letters = [ch for ch in pattern if ch.isalpha() and ch.isupper()]
    if not letters:
        return 0
    return (ord(max(letters)) - ord('A') + 1)

# ──────────────────────────────────────────────────────────────
# 2.  Recursively build *all* (value, expr) pairs for a tuple
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def all_exprs(nums: tuple[int, ...]) -> set[tuple[Fraction, str]]:
    """
    Return a set of (value, expr-string) pairs for the multiset `nums`.

    `nums` is a tuple so it can contain duplicates and be hashed for caching.
    """
    # Base case – one number left
    if len(nums) == 1:
        n = Fraction(nums[0])
        # print integers cleanly; fractions stay in (a/b) form
        lit = str(n.numerator) if n.denominator == 1 else f'({n})'
        return {(n, lit)}

    results = set()

    # Split the tuple at every possible cut point
    for k in range(1, len(nums)):
        left_nums  = nums[:k]
        right_nums = nums[k:]

        # Recurse for every expression on each side
        for vL, sL in all_exprs(left_nums):
            for vR, sR in all_exprs(right_nums):

                # Apply every operator
                for sym, fn in OPS.items():

                    # Left op Right  (always allowed)
                    val = fn(vL, vR)
                    if val is not None:
                        results.add((val, f'({sL}{sym}{sR})'))

                    # Right op Left  (needed for - and / only)
                    if sym in ('-', '/'):
                        val = fn(vR, vL)
                        if val is not None:
                            results.add((val, f'({sR}{sym}{sL})'))

    del vL, sL, vR, sR, sym, fn, val, left_nums, right_nums
    return results

# ──────────────────────────────────────────────────────────────
# 3.  Public API – search every permutation of the numbers
# ──────────────────────────────────────────────────────────────
"""
Exhaustive solver for the Countdown game.

Input
-----
    nums   : list[int]        # any integers 1-99
    target : int | Fraction   # goal value 1-99

Output
------
    List of fully-parenthesised infix strings, each of which

        • uses every number exactly once,
        • employs only + – * /,
        • equals the target under exact rational arithmetic.

Example
-------
    >>> from utils import solve
    >>> for s in solve([19, 36, 55, 7], 65):
    ...     print(s)
    (55-19)+(36-7)
"""
def solve(numbers: list[int], target) -> list[str]:
    """
    Return *all* expressions (as strings) that evaluate to `target`.
    """
    target = Fraction(target)
    solutions = set()

    for perm in permutations(numbers):
        for val, expr in all_exprs(tuple(perm)):
            if val == target:
                # strip outermost parentheses for readability
                solutions.add(expr[1:-1] if expr.startswith('(') else expr)

    return sorted(solutions, key=len)   # shortest first

# ──────────────────────────────────────────────────────────────
# 4.  Public API – from a model output, extract the list of all patterns used
# ──────────────────────────────────────────────────────────────
def extract_patterns(text: str, nums: List[int]) -> List[str]:
    """
    Return every arithmetic expression inside *text* that

      • uses **each** integer in `nums` exactly once (multiset‑wise)
      • contains **no other numbers**
      • is made only of + – * /, parentheses and spaces.

    Each result is a *canonical*, fully‑parenthesised string where the numbers
    are renamed A, B, C… according to their first appearance order.
    Duplicate canonical strings are removed while preserving encounter order.
    """

    # ---------- 1 load all expressions and their canonical forms  ----------
    pattern_mapping = {}
    with open("annotated_expressions.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = json.loads(line)
            pattern_mapping.update(mapping)

    # ---------- 2 extract raw candidate substrings ----------
    nums_as_str   = [str(n) for n in nums]
    must_have     = collections.Counter(nums_as_str)
    allowed_set   = set(nums_as_str)
    ARITHM_RE = re.compile(r'(?:[0-9]+|\(|\)|[+\-*/]|[ \t])+') # potential update: r'^(?!.*[0-9]+[ \t]+[0-9])(?:[0-9]+|\(|\)|[+\-*/]|[ \t])+$'
    raw_exprs = []

    for m in ARITHM_RE.finditer(text):
        seg = m.group(0).strip()
        if not seg:
            continue
        # strip bullet dashes like "- 92 - 26 ..."
        if seg.startswith('-') and (len(seg) == 1 or not seg[1].isdigit() and seg[1] != '('):
            seg = seg[1:].lstrip()
        if not seg or not re.search(r'[+\-*/]', seg):
            continue                                               # need an operator
        seg_nums = re.findall(r'\d+', seg)
        if collections.Counter(seg_nums) == must_have and set(seg_nums) <= allowed_set:
            raw_exprs.append(seg)

    # ---------- 3 helpers: renumber → postfix → tree → string ----------
    def renumber(expr: str) -> str:
        # mapping, nxt = {}, ord('A')
        nxt = ord('A')
        def repl(_):
            nonlocal nxt
            letter = chr(nxt)
            nxt += 1
            return letter
        return re.sub(r'\d+', repl, expr)
    
    TOK = lambda s: [c for c in s if c.strip()]
    prec = {'+': 1, '-': 1, '*': 2, '/': 2}

    def to_postfix(tokens):
        out, op = [], []
        for t in tokens:
            if t.isalpha():
                out.append(t)
            elif t in prec:
                while op and op[-1] in prec and prec[op[-1]] >= prec[t]:
                    out.append(op.pop())
                op.append(t)
            elif t == '(':
                op.append(t)
            elif t == ')':
                while op and op[-1] != '(':
                    out.append(op.pop())
                if not op: raise ValueError('mismatched )')
                op.pop()
        while op:
            top = op.pop()
            if top == '(': raise ValueError('mismatched (')
            out.append(top)
        return out
    
    class Node:
        __slots__ = ('v', 'l', 'r')
        def __init__(self, v, l=None, r=None):
            self.v, self.l, self.r = v, l, r

    def build_tree(post):
        stk = []
        for t in post:
            if t.isalpha():
                stk.append(Node(t))
            else:
                if len(stk) < 2:
                    raise ValueError('malformed expression')
                r, l = stk.pop(), stk.pop()
                stk.append(Node(t, l, r))
        if len(stk) != 1:
            raise ValueError('malformed expression')
        return stk[0]
    
    def stringify(n: Node, outer=True):
        if n.l is None:
            return n.v
        s = f"{stringify(n.l, False)}{n.v}{stringify(n.r, False)}"
        return s if outer else f"({s})"
    
    # ---------- 4 canonicalise & de‑duplicate ----------
    patterns, canonical_patterns, seen_can = [], [], set()
    for raw in raw_exprs:
        expr = re.sub(r'\s+', '', raw)
        try:
            pattern = stringify(build_tree(to_postfix(TOK(renumber(expr)))))
            canonical_pattern = pattern_mapping[pattern]
        except ValueError:                     # skip malformed ones
            continue
        puzzle_size = get_size(pattern)
        if puzzle_size != len(nums): continue  # skip wrong size

        patterns.append(pattern)
        canonical_patterns.append(canonical_pattern)
        seen_can.add(canonical_pattern)
    return patterns, canonical_patterns, seen_can

def extract_patterns_reverse(text: str, nums: List[int]) -> List[str]:
    """
    Return every arithmetic expression inside *text* that

      • uses **each** integer in `nums` exactly once (multiset‑wise)
      • contains **no other numbers**
      • is made only of + – * /, parentheses and spaces.

    Each result is a *canonical*, fully‑parenthesised string where the numbers
    are renamed A, B, C… according to their first appearance order.
    Duplicate canonical strings are removed while preserving encounter order.
    """

    # ---------- 1 load all expressions and their canonical forms  ----------
    pattern_mapping = {}
    with open("annotated_expressions_reverse.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = json.loads(line)
            pattern_mapping.update(mapping)

    # ---------- 2 extract raw candidate substrings ----------
    nums_as_str   = [str(n) for n in nums]
    must_have     = collections.Counter(nums_as_str)
    allowed_set   = set(nums_as_str)
    ARITHM_RE = re.compile(r'(?:[0-9]+|\(|\)|[+\-*/]|[ \t])+') # potential update: r'^(?!.*[0-9]+[ \t]+[0-9])(?:[0-9]+|\(|\)|[+\-*/]|[ \t])+$'
    raw_exprs = []

    for m in ARITHM_RE.finditer(text):
        seg = m.group(0).strip()
        if not seg:
            continue
        # strip bullet dashes like "- 92 - 26 ..."
        if seg.startswith('-') and (len(seg) == 1 or not seg[1].isdigit() and seg[1] != '('):
            seg = seg[1:].lstrip()
        if not seg or not re.search(r'[+\-*/]', seg):
            continue                                               # need an operator
        seg_nums = re.findall(r'\d+', seg)
        if collections.Counter(seg_nums) == must_have and set(seg_nums) <= allowed_set:
            raw_exprs.append(seg)

    # ---------- 3 helpers: renumber → postfix → tree → string ----------
    def renumber(expr: str) -> str:
        # mapping, nxt = {}, ord('A')
        nxt = ord('A')
        def repl(_):
            nonlocal nxt
            letter = chr(nxt)
            nxt += 1
            return letter
        return re.sub(r'\d+', repl, expr)
    
    TOK = lambda s: [c for c in s if c.strip()]
    prec = {'+': 1, '-': 1, '*': 2, '/': 2}

    def to_postfix(tokens):
        out, op = [], []
        for t in tokens:
            if t.isalpha():
                out.append(t)
            elif t in prec:
                while op and op[-1] in prec and prec[op[-1]] >= prec[t]:
                    out.append(op.pop())
                op.append(t)
            elif t == '(':
                op.append(t)
            elif t == ')':
                while op and op[-1] != '(':
                    out.append(op.pop())
                if not op: raise ValueError('mismatched )')
                op.pop()
        while op:
            top = op.pop()
            if top == '(': raise ValueError('mismatched (')
            out.append(top)
        return out
    
    class Node:
        __slots__ = ('v', 'l', 'r')
        def __init__(self, v, l=None, r=None):
            self.v, self.l, self.r = v, l, r

    def build_tree(post):
        stk = []
        for t in post:
            if t.isalpha():
                stk.append(Node(t))
            else:
                if len(stk) < 2:
                    raise ValueError('malformed expression')
                r, l = stk.pop(), stk.pop()
                stk.append(Node(t, l, r))
        if len(stk) != 1:
            raise ValueError('malformed expression')
        return stk[0]
    
    def stringify(n: Node, outer=True):
        if n.l is None:
            return n.v
        s = f"{stringify(n.l, False)}{n.v}{stringify(n.r, False)}"
        return s if outer else f"({s})"
    
    # ---------- 4 canonicalise & de‑duplicate ----------
    patterns, canonical_patterns, seen_can = [], [], set()
    for raw in raw_exprs:
        expr = re.sub(r'\s+', '', raw)
        try:
            pattern = stringify(build_tree(to_postfix(TOK(renumber(expr)))))
            canonical_pattern = pattern_mapping[pattern]
        except ValueError:                     # skip malformed ones
            continue
        puzzle_size = get_size(pattern)
        if puzzle_size != len(nums): continue  # skip wrong size

        patterns.append(pattern)
        canonical_patterns.append(canonical_pattern)
        seen_can.add(canonical_pattern)
    return patterns, canonical_patterns, seen_can

if __name__ == "__main__":

    # ------------------------------------------------------------------
    #                       ✅  UNIT TEST SUITE
    # ------------------------------------------------------------------
    tests = [
        {
            "name": "sample_from_prompt",
            "text": ("Let's try: 98 - 34 - 96 + 1. "
                    "This gives me 98 - 34 = 64, then 64 - 96 = -32. "
                    "That's not 33. "
                    "Let's try: 34 + 98 - 96 + 1. "
                    "This gives me 34 + 98 = 132, then 132 - 96 = 36, and finally 36 + 1 = 37. "
                    "Let's try: 98 - 34 - 96 - 1."),
            "nums": [98, 34, 96, 1],
            "expect": ["((A-B)-C)+D", "((A+B)-C)+D", "((A-B)-C)-D"]
        },
        {
            "name": "with_parentheses",
            "text": "Result: 92 / (26 - 12) + 3 is something.",
            "nums": [92, 26, 12, 3],
            "expect": ["(A/(B-C))+D"]
        },
        {
            "name": "explicit_parentheses_multiplication",
            "text": "Try 3*(5+2)-7 now.",
            "nums": [3, 5, 2, 7],
            "expect": ["(A*(B+C))-D"]
        },
        {
            "name": "division_associativity_variants",
            "text": "Multiple: 8/4/2+1 and 8/(4/2)+1 maybe weird.",
            "nums": [8, 4, 2, 1],
            "expect": ["((A/B)/C)+D", "(A/(B/C))+D"]
        }
    ]

    for t in tests:
        patterns, unique_patterns = extract_patterns(t["text"], t["nums"])
        if sorted(patterns) != sorted(t["expect"]):
            print(f"❌ {t['name']} failed:\n   expected {t['expect']}\n   got      {patterns}")
        else:
            print(f"✅ {t['name']} passed")