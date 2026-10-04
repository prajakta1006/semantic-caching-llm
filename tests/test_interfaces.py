"""Tests for shared interface contracts."""

from src.interfaces import (
    CacheInterface,
    EvaluatorInterface,
    LLMInterface,
    QueryPreprocessorInterface,
)
from src.llm import MockLLM
from src.pipeline import DefaultPreprocessor


def test_preprocessor_interface():

    preprocessor = DefaultPreprocessor()

    assert isinstance(
        preprocessor,
        QueryPreprocessorInterface,
    )

    assert (
        preprocessor.preprocess(
            "  hello  "
        )
        == "hello"
    )


def test_llm_interface():

    llm = MockLLM()

    assert isinstance(
        llm,
        LLMInterface,
    )


def test_cache_interface_is_abstract():

    assert CacheInterface.__abstractmethods__ == {
        "get",
        "put",
        "size",
    }


def test_evaluator_interface_is_abstract():

    assert EvaluatorInterface.__abstractmethods__ == {
        "record",
        "get_metrics",
    }
