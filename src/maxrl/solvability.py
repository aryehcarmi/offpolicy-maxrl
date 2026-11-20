"""Exhaustive witness search for the binary Countdown verifier."""
import operator
from collections.abc import Sequence
from fractions import Fraction
from itertools import permutations


_OPERATIONS = (
    ("+", 1, operator.add),
    ("-", 1, operator.sub),
    ("*", 2, operator.mul),
    ("/", 2, operator.truediv),
)


def find_solution(numbers: Sequence[int], target: int) -> str | None:
    """Return an arithmetic expression using every input once, or None.

    Every binary expression tree can be reduced by combining two siblings.
    Trying every ordered pair and operation therefore covers all trees. Equal
    inputs occupy separate positions, and a match counts only at the root.
    """
    # The verifier only accepts unsigned integer literals as leaves.
    if not numbers or any(number < 0 for number in numbers):
        return None

    def search(terms):
        if len(terms) == 1:
            value, text, _ = terms[0]
            return text if value == target and len(text) <= 256 else None

        for i, j in permutations(range(len(terms)), 2):
            left, left_text, left_precedence = terms[i]
            right, right_text, right_precedence = terms[j]
            remaining = [term for k, term in enumerate(terms) if k != i and k != j]
            for symbol, precedence, operation in _OPERATIONS:
                if symbol == "/" and right == 0:
                    continue
                # Binary operators associate to the left at equal precedence.
                lhs = f"({left_text})" if left_precedence < precedence else left_text
                rhs = f"({right_text})" if right_precedence <= precedence else right_text
                expression = f"{lhs}{symbol}{rhs}"
                combined = (operation(left, right), expression, precedence)
                witness = search([*remaining, combined])
                if witness is not None:
                    return witness
        return None

    terms = [(Fraction(number), str(number), 3) for number in numbers]
    return search(terms)
