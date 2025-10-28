from itertools import product
import re
import json

ops = ['+', '-', '*', '/']

def all_expressions(k):

    if k == 6:
        shapes = [
            "A{0}(B{1}(C{2}(D{3}(E{4}F))))",
            "A{0}(B{1}(C{2}((D{4}E){3}F)))",
            "A{0}(B{1}((C{3}D){2}(E{4}F)))",
            "A{0}(B{1}((C{3}(D{4}E)){2}F))",
            "A{0}(B{1}(((C{4}D){3}E){2}F))",
            "A{0}((B{2}C){1}(D{3}(E{4}F)))",
            "A{0}((B{2}C){1}((D{4}E){3}F))",
            "A{0}((B{2}(C{3}D)){1}(E{4}F))",
            "A{0}(((B{3}C){2}D){1}(E{4}F))",
            "A{0}((B{2}(C{3}(D{4}E))){1}F)",
            "A{0}((B{2}((C{4}D){3}E)){1}F)",
            "A{0}(((B{3}C){2}(D{4}E)){1}F)",
            "A{0}(((B{3}(C{4}D)){2}E){1}F)",
            "A{0}((((B{4}C){3}D){2}E){1}F)",

            "(A{1}B){0}(C{2}(D{3}(E{4}F)))",
            "(A{1}B){0}(C{2}((D{4}E){3}F))",
            "(A{1}B){0}((C{3}D){2}(E{4}F))",
            "(A{1}B){0}((C{3}(D{4}E)){2}F)",
            "(A{1}B){0}(((C{4}D){3}E){2}F)",

            "((A{2}B){1}C){0}((D{4}E){3}F)",
            "((A{2}B){1}C){0}(D{3}(E{4}F))",
            "(A{1}(B{2}C)){0}((D{4}E){3}F)",
            "(A{1}(B{2}C)){0}(D{3}(E{4}F))",

            "((A{2}B){1}(C{3}D)){0}(E{4}F)",
            "((A{2}(B{3}C)){1}D){0}(E{4}F)",
            "(((A{3}B){2}C){1}D){0}(E{4}F)",
            "(A{1}(B{2}(C{3}D))){0}(E{4}F)",
            "(A{1}((B{3}C){2}D)){0}(E{4}F)",

            "((A{2}B){1}(C{3}(D{4}E))){0}F",
            "((A{2}B){1}((C{4}D){3}E)){0}F",
            "((A{2}(B{3}C)){1}(D{4}E)){0}F",
            "(((A{3}B){2}C){1}(D{4}E)){0}F",
            "((A{2}(B{3}(C{4}D))){1}E){0}F",
            "((A{2}((B{4}C){3}D)){1}E){0}F",
            "(((A{3}B){2}(C{4}D)){1}E){0}F",
            "(((A{3}(B{4}C)){2}D){1}E){0}F",
            "((((A{4}B){3}C){2}D){1}E){0}F",
            "(A{1}(B{2}(C{3}(D{4}E)))){0}F",
            "(A{1}(B{2}((C{4}D){3}E))){0}F",
            "(A{1}((B{3}C){2}(D{4}E))){0}F",
            "(A{1}((B{3}(C{4}D)){2}E)){0}F",
            "(A{1}(((C{4}D){3}E){2}F)){0}F", 
        ]
        for template in shapes:
            for op1, op2, op3, op4, op5 in product(ops, repeat=5):
                yield template.format(op1, op2, op3, op4, op5)

    if k == 5:
        shapes = [
            "A{0}(B{1}(C{2}(D{3}E)))",
            "A{0}(B{1}((C{3}D){2}E))",
            "A{0}((B{2}C){1}(D{3}E))",
            "A{0}((B{2}(C{3}D)){1}E)",
            "A{0}(((B{3}C){2}D){1}E)",
            "(A{1}B){0}(C{2}(D{3}E))",
            "(A{1}B){0}((C{3}D){2}E)",
            "(A{1}(B{2}C)){0}(D{3}E)",
            "((A{2}B){1}C){0}(D{3}E)",
            "(A{1}(B{2}(C{3}D))){0}E",
            "(A{1}((B{3}C){2}D)){0}E",
            "((A{2}B){1}(C{3}D)){0}E",
            "((A{2}(B{3}C)){1}D){0}E",
            "(((A{3}B){2}C){1}D){0}E",
        ]
        for template in shapes:
            for op1, op2, op3, op4 in product(ops, repeat=4):
                yield template.format(op1, op2, op3, op4)

    if k == 4:
        shapes = [
            "A{0}(B{1}(C{2}D))",
            "A{0}((B{2}C){1}D)",
            "(A{1}B){0}(C{2}D)",
            "(A{1}(B{2}C)){0}D",
            "((A{2}B){1}C){0}D",
        ]
        for template in shapes:
            for op1, op2, op3 in product(ops, repeat=3):
                yield template.format(op1, op2, op3)

    if k == 3:
        shapes = [
            "(A{0}B){1}C",
            "A{0}(B{1}C)"
        ]
        for op1, op2 in product(ops, repeat=2):
            for template in shapes:
                yield template.format(op1, op2)

    if k == 2:
        shapes = [
            "A{0}B"
        ]
        for op1 in ops:
            for template in shapes:
                yield template.format(op1)

class ExpressionTree:
    """
    Represents a node in an expression tree.
    This class is used to build a tree structure from an arithmetic expression,
    which is then transformed into a canonical form.
    """
    def __init__(self, op, children=None):
        self.op = op
        self.children = children if children is not None else []
        self.sign = '+'
        self.weight = 1 if not children else sum(c.weight for c in children)

    def __repr__(self):
        return f"ExpressionTree(op='{self.op}', sign='{self.sign}', weight={self.weight}, children={self.children})"

def parse_expression(expression):
    """
    Parses an arithmetic expression string into a binary expression tree.
    It handles parentheses and basic operator precedence.
    """
    expression = expression.strip()
    
    # Base case: if the expression is a single variable
    if re.match(r'^[A-Z]$', expression):
        return ExpressionTree(expression)

    # Handle parentheses
    if expression.startswith('(') and expression.endswith(')'):
        balance = 0
        for i, char in enumerate(expression):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            if balance == 0 and i < len(expression) - 1:
                break
        else: # if the loop completes without breaking
            return parse_expression(expression[1:-1])

    # Find the operator with the lowest precedence (+ or -)
    balance = 0
    for i in range(len(expression) - 1, -1, -1):
        char = expression[i]
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        elif balance == 0 and char in "+-":
            left = parse_expression(expression[:i])
            right = parse_expression(expression[i+1:])
            return ExpressionTree(char, [left, right])

    # Find the operator with the next lowest precedence (* or /)
    balance = 0
    for i in range(len(expression) - 1, -1, -1):
        char = expression[i]
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        elif balance == 0 and char in "*/":
            left = parse_expression(expression[:i])
            right = parse_expression(expression[i+1:])
            return ExpressionTree(char, [left, right])

    raise ValueError(f"Invalid expression: {expression}")


def build_generic_tree(node):
    """
    Converts a binary expression tree into a generic, signed tree by merging
    chains of same-precedence operators.
    """
    if not node.children:
        return node

    # Recursively process children
    node.children = [build_generic_tree(c) for c in node.children]

    # Set signs for subtraction and division
    if node.op in ['-', '/']:
        node.children[1].sign = '-'

    # Merge chains of same-precedence operators
    if node.op in ['+', '-']:
        new_children = []
        for child in node.children:
            if child.op in ['+', '-']:
                for grandchild in child.children:
                    if child.sign == '-':
                        grandchild.sign = '+' if grandchild.sign == '-' else '-'
                    new_children.append(grandchild)
            else:
                new_children.append(child)
        node.children = new_children
        node.op = '+' # Canonical operator for additive group
    elif node.op in ['*', '/']:
        new_children = []
        for child in node.children:
            if child.op in ['*', '/']:
                for grandchild in child.children:
                    if child.sign == '-':
                        grandchild.sign = '+' if grandchild.sign == '-' else '-'
                    new_children.append(grandchild)
            else:
                new_children.append(child)
        node.children = new_children
        node.op = '*' # Canonical operator for multiplicative group

    # Sort children based on sign and then weight
    node.children.sort(key=lambda x: (x.sign, -x.weight, -sum(1 for gc in x.children if gc.sign == '+')))
    
    # Update weight
    node.weight = sum(c.weight for c in node.children)

    return node

def get_canonical_form(tree, parent_op=None, symbols_map=None):
    """
    Generates the canonical string representation from the generic, sorted tree.
    It renames variables and handles parentheses.
    """
    if symbols_map is None:
        symbols_map = {}

    if not tree.children:
        if tree.op not in symbols_map:
            symbols_map[tree.op] = chr(ord('A') + len(symbols_map))
        return symbols_map[tree.op]

    parts = []
    for i, child in enumerate(tree.children):
        op = tree.op
        if child.sign == '-':
            op = '+' if op == '*' else '-' if op == '+' else '/' if op == '*' else '*'

        # Determine if parentheses are needed
        needs_parens = (
            child.weight > 1 and
            ((tree.op == '*' and child.op == '+') or (tree.op == '/' and child.op == '+'))
        )
        
        child_expr = get_canonical_form(child, tree.op, symbols_map)
        
        if needs_parens:
            child_expr = f"({child_expr})"
        
        if i > 0:
            parts.append(op if child.sign == '+' else ('-' if tree.op == '+' else '/'))
        
        parts.append(child_expr)

    return "".join(parts)


def convert_to_canonical(expression):
    """
    Main function to convert an expression to its canonical form.
    """
    binary_tree = parse_expression(expression)
    generic_tree = build_generic_tree(binary_tree)
    return get_canonical_form(generic_tree)

def run_tests():
    """
    Runs the unit tests based on the provided annotated_expressions.txt file.
    """
    passed = 0
    failed = 0
    with open('annotated_expressions.txt', 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                for input_expr, expected_output in data.items():
                    try:
                        actual_output = convert_to_canonical(input_expr)
                        if actual_output == expected_output:
                            # print(f"PASS: {input_expr} -> {actual_output}")
                            passed += 1
                        else:
                            print(f"FAIL: {input_expr}")
                            print(f"  Expected: {expected_output}")
                            print(f"  Got:      {actual_output}")
                            failed += 1
                    except Exception as e:
                        print(f"ERROR processing '{input_expr}': {e}")
                        failed += 1
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()}")
                continue # Ignore empty or malformed lines
    
    print("\n--- Test Summary ---")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("--------------------")


if __name__ == "__main__":
    data = dict()
    all_lines = []
    for puzzle_size in range(2, 7):
        lines = []
        for i, expr in enumerate(all_expressions(k=puzzle_size), 1):
            lines.append("{\"" + f"{expr}" + "\": \"" + convert_to_canonical(expr) + "\"}")

        all_lines.append("\n".join(lines))

        patterns = []
        unique_patterns = set()
        for line in lines:
            mapping = json.loads(line)
            k, v = list(mapping.items())[0]
            if v not in unique_patterns:
                patterns.append(v)
                unique_patterns.add(v)

        data_ = []
        for line in lines:
            mapping = json.loads(line)
            k, v = list(mapping.items())[0]
            pattern_idx = patterns.index(v)
            data_.append({"expression": k, "canonical": v, "pattern_idx": pattern_idx})

        data[str(puzzle_size)] = data_

    with open(f"annotated_expressions.txt", "w") as f:
        f.write("\n".join(all_lines))

    with open("annotated_expressions.json", "w") as f:
        json.dump(data, f, indent=2)