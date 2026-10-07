import pytest

from agentflow.calculator import CalculationError, extract_expression, numbers_in, safe_eval


def test_extracts_labelled_expression():
    assert extract_expression("Expression: (5829 - 5735)") == "(5829 - 5735)"
    assert extract_expression("Let me think.\nExpression: 412 / 380") == "412 / 380"


def test_cleans_money_commas_and_trailing_result():
    assert extract_expression("Expression: ($5,829 - $5,735) / $5,735 = 0.0164") == "(5829 - 5735) / 5735"
    assert extract_expression("`35 + 41`") == "35 + 41"
    assert extract_expression("Expression: 12% * 300") == "12 * 300"


def test_rejects_words_and_empty_output():
    assert extract_expression("Expression: revenue - cost") is None
    assert extract_expression("I cannot answer") is None
    assert extract_expression("") is None
    assert extract_expression(None) is None


def test_safe_eval_arithmetic():
    assert safe_eval("(5829 - 5735)") == 94
    assert safe_eval("(412 - 380) / 380") == pytest.approx(0.0842105)
    assert safe_eval("-3 + 2 * 4") == 5
    assert safe_eval("1.1 ** 2") == pytest.approx(1.21)


def test_safe_eval_refuses_code_and_bad_math():
    for bad in ("__import__('os')", "a + 1", "10 / 0", "2 ** 100", "(1 +"):
        with pytest.raises(CalculationError):
            safe_eval(bad)


def test_numbers_in():
    assert numbers_in("(5829 - 5735) / 5735") == [5829.0, 5735.0, 5735.0]
    assert numbers_in("0.5 * .25") == [0.5, 0.25]