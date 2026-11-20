from itertools import permutations, product

import pytest

from maxrl.solvability import find_solution
from maxrl.verifier import verify


@pytest.mark.parametrize(("numbers", "target"), [
    ([3, 3, 8, 8], 24),  # 8 / (3 - 8 / 3) requires a fractional intermediate.
    ([2, 7, 8], -40),  # (2 - 7) * 8 requires a negative intermediate.
    ([2, 3, 4, 5], 49),  # (2 + 5) * (3 + 4) requires two compound operands.
    ([3, 3, 5], 0),
    ([10**253, 1], 10**253 + 1),  # The unparenthesized sum is exactly 256 characters.
])
def test_witness_passes_verifier(numbers, target):
    expression = find_solution(numbers, target)
    assert expression is not None
    assert verify(f"<answer>{expression}</answer>", numbers, target)


@pytest.mark.parametrize(("numbers", "target"), [
    ([2, 3, 8], 5),  # Using only the first two numbers would give the target.
    ([2, 2, 2], 13),  # The search can encounter a zero denominator.
    ([2**53 + 1, 1, 1], 2**53 - 2),  # Rounding the first number could give a false match.
    ([10**254, 1], 10**254 + 1),  # Every expression exceeds the verifier's length limit.
])
def test_unsolvable(numbers, target):
    assert find_solution(numbers, target) is None


def test_three_number_search_agrees_with_every_verifier_expression():
    numbers = [4, 2, 3]
    answers = []
    for a, b, c in permutations(numbers):
        for op1, op2 in product("+-*/", repeat=2):
            answers.append(f"<answer>({a}{op1}{b}){op2}{c}</answer>")
            answers.append(f"<answer>{a}{op1}({b}{op2}{c})</answer>")

    for target in range(-25, 26):
        expected = any(verify(answer, numbers, target) for answer in answers)
        expression = find_solution(numbers, target)
        assert (expression is not None) == expected, target
        if expression is not None:
            assert verify(f"<answer>{expression}</answer>", numbers, target)
