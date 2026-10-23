from src.rag_app.config import config
from src.rag_app.rag_pipeline import RAGPipeline
from scripts.evaluate import deterministic_mock_llm


def test_unsupported_weather_question():
    """Verify that an unsupported question regarding weather produces exactly 'Not found'."""
    pipeline = RAGPipeline(config)
    res = pipeline.run("What was the weather in Delhi yesterday?", mock_llm=deterministic_mock_llm)

    assert res.is_answered is False
    assert res.answer == "Not found"
    assert len(res.citations) == 0
    assert res.formatted_display() == "Not found"


def test_unsupported_pricing_question():
    """Verify that an unsupported commercial pricing question produces exactly 'Not found'."""
    pipeline = RAGPipeline(config)
    res = pipeline.run("What is the price of a MacBook Air?", mock_llm=deterministic_mock_llm)

    assert res.is_answered is False
    assert res.answer == "Not found"
    assert len(res.citations) == 0
    assert res.formatted_display() == "Not found"


def test_unsupported_movie_copyright_penalty():
    """Verify that external film copyright penalty question produces exactly 'Not found'."""
    pipeline = RAGPipeline(config)
    res = pipeline.run("What is the penalty for copyright infringement of a movie under Indian law?", mock_llm=deterministic_mock_llm)

    assert res.is_answered is False
    assert res.answer == "Not found"
    assert len(res.citations) == 0
    assert res.formatted_display() == "Not found"


def test_service_error_distinction_from_not_found():
    """Verify that operational errors (e.g., uninitialized index) are distinct from factual 'Not found'."""
    from src.rag_app.config import AppConfig
    from pathlib import Path

    # Config pointing to non-existent chroma directory
    invalid_cfg = AppConfig(chroma_dir=Path("/non/existent/path/chroma_db"))
    pipeline = RAGPipeline(invalid_cfg)
    res = pipeline.run("What is an ordinary application?")

    assert res.is_error is True
    assert res.answer != "Not found"
    assert "not initialized" in res.error_message
