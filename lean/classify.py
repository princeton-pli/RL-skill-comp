"""Classify a Lean 4 proof by the structure of its top-level binary split.

A proof is classified only if its body opens (at the top level) with a tactic
that creates exactly two subgoals:

    constructor / fconstructor
    refine ⟨?_, ?_⟩
    apply Iff.intro / apply And.intro   (also `exact ...`)
    refine Iff.intro ?_ ?_ / refine And.intro ?_ ?_

The two subgoals must then be discharged in two `·`-bulleted branches. We count
the non-blank, non-comment lines in each branch (w_L, w_R), so a nested `by`
block inside a `have` counts towards the branch it belongs to, set
r = w_L / (w_L + w_R), and bucket the proof as

    left-heavy  (L)  if r >= left_thr   (default 0.6)
    right-heavy (R)  if r <= right_thr  (default 0.4)
    balanced    (B)  otherwise.

Any other proof (no top-level binary split, or not exactly two bulleted
branches) is labelled `other` and excluded from the L/B/R distribution.
"""
import re

# ---------------------------------------------------------------------------
# Extracting the proof body from a model response / SFT target
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r'^```(lean4?|tactics?)\s*\n(.*?)^```\s*$',
    re.DOTALL | re.MULTILINE,
)


def parse_proof_body(response: str) -> str:
    """Return the proof body (the text after `:= by`) of a whole-proof response.

    (i) Locate all fenced code blocks tagged lean / lean4 / tactic / tactics;
    (ii) prefer the last lean* block over any tactics* scratch block;
    (iii) take the substring after the first `:= by` in the chosen block, i.e.
    the proof of the outer theorem rather than of an inner `have ... := by`.
    Returns '' if no proof body can be found.
    """
    matches = _FENCE_RE.findall(response)
    if matches:
        lean_blocks = [content for tag, content in matches if tag.startswith('lean')]
        candidate = lean_blocks[-1] if lean_blocks else matches[-1][1]
    else:
        # No complete fenced block: keep everything up to a closing fence.
        end = response.find('\n```')
        candidate = response[:end] if end != -1 else response

    m = re.search(r':=\s*by\b', candidate)
    if m is None:
        return ''
    body = candidate[m.end():]
    if body.startswith('\n'):
        body = body[1:]
    return body.rstrip()


# ---------------------------------------------------------------------------
# Classifying a proof body
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(
    r'^\s*(constructor|fconstructor|refine\s+[⟨⟪].*?[⟩⟫]|'
    r'(?:apply|exact)\s+(?:Iff|And)\.intro|'
    r'refine\s+(?:Iff|And)\.intro\s+[?_]+\s+[?_]+)\s*$',
)

_BULLET_RE = re.compile(r'^(\s*)·\s')


def _is_blank_or_comment(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith('--')


def find_split_line(lines):
    """Index of the first line that is a binary-split tactic, or None."""
    for i, line in enumerate(lines):
        if _is_blank_or_comment(line):
            continue
        if _SPLIT_RE.match(line):
            return i
    return None


def branch_weights(lines, split_idx):
    """Return (w_L, w_R) for the two `·` branches after the split, or None."""
    branches = []
    current = None
    bullet_indent = None  # indentation of the first bullet = branch level

    for line in lines[split_idx + 1:]:
        if _is_blank_or_comment(line):
            continue
        bm = _BULLET_RE.match(line)
        if bm:
            indent = len(bm.group(1))
            if bullet_indent is None:
                bullet_indent = indent
            if indent == bullet_indent:  # a new branch starts
                if current is not None:
                    branches.append(current)
                current = [line[indent + 2:].rstrip()]  # text after `· `
                continue
        line_indent = len(line) - len(line.lstrip())
        if bullet_indent is not None and line_indent <= bullet_indent and not bm:
            break  # dedented past the branches: the split region is over
        if current is not None:
            current.append(line.rstrip())

    if current is not None:
        branches.append(current)
    if len(branches) != 2:
        return None
    return tuple(sum(1 for l in br if not _is_blank_or_comment(l)) for br in branches)


def classify(proof_body: str, left_thr: float = 0.6, right_thr: float = 0.4):
    """Return 'L', 'B', 'R', or 'other' for a proof body."""
    lines = proof_body.split('\n')
    split_idx = find_split_line(lines)
    if split_idx is None:
        return 'other'
    weights = branch_weights(lines, split_idx)
    if weights is None or sum(weights) == 0:
        return 'other'
    r = weights[0] / sum(weights)
    if r >= left_thr:
        return 'L'
    if r <= right_thr:
        return 'R'
    return 'B'


def classify_response(response: str, left_thr: float = 0.6, right_thr: float = 0.4):
    """Parse a full response (model output or SFT target) and classify it."""
    body = parse_proof_body(response)
    if not body.strip():
        return 'other'
    return classify(body, left_thr, right_thr)


# Threshold settings reported in the paper.
THRESHOLDS = {
    'main (Table 3)': (0.6, 0.4),
    'alt (Table 12)': (2 / 3, 1 / 3),
}


def distribution(responses, left_thr=0.6, right_thr=0.4):
    """Counts of L/B/R/other over an iterable of responses."""
    counts = {'B': 0, 'L': 0, 'R': 0, 'other': 0}
    for resp in responses:
        counts[classify_response(resp, left_thr, right_thr)] += 1
    return counts


def format_distribution(counts):
    n = counts['B'] + counts['L'] + counts['R']
    if n == 0:
        return 'no classifiable proofs'
    return (f"balanced {100 * counts['B'] / n:.1f}%  "
            f"left-heavy {100 * counts['L'] / n:.1f}%  "
            f"right-heavy {100 * counts['R'] / n:.1f}%  "
            f"(classifiable {n}, other {counts['other']})")
