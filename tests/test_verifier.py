import pytest

from maxrl.verifier import verify


@pytest.mark.parametrize(("text", "numbers", "target"), [
    ("I can group the terms.\n<answer>(2 + 3) * 4</answer>", [2, 3, 4], 20),
    ("<answer>3/(1-3/4)</answer>", [1, 3, 3, 4], 12),
    ("<answer>(1-5)*(2-4)</answer>", [1, 2, 4, 5], 8),
    ("<answer>(2+2)/4</answer>\n", [2, 2, 4], 1),
])
def test_valid(text, numbers, target):
    assert verify(text, numbers, target)


@pytest.mark.parametrize("expression", [
    "9", "1+4+4", "2+3", "2+3+4+0", "2.0+3+4", "2**3+4", "2//3+4",
    "2+3*4", "(2+3+4", "+2+3+4", "2 + int('3') + 4", "02+3+4", "2⁺3+4",
    "2+3+4=9", "2+3+4 #ok", "2,3,4",
])
def test_reject_expression(expression):
    assert not verify(f"<answer>{expression}</answer>", [2, 3, 4], 9)


@pytest.mark.parametrize("text", [
    "2+3+4", "<answer>2+3+4", "<answer>9</answer><answer>2+3+4</answer>",
    "<answer>2+3+4</answer> or something else",
])
def test_require_single_final_answer(text):
    assert not verify(text, [2, 3, 4], 9)


def test_zero_division():
    assert not verify("<answer>4/(2-2)</answer>", [2, 2, 4], 0)
