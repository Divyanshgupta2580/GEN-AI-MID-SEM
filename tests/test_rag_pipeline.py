import os
import pytest
from src.rag_app.config import config
from src.rag_app.rag_pipeline import RAGPipeline
from scripts.evaluate import deterministic_mock_llm


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline(config)


def test_q1_design_definition(pipeline):
    """Q1: Verify statutory definition of design."""
    q = "What does the manual mean by the term 'design'?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)

    assert res.is_answered is True
    assert "shape" in res.answer.lower()
    assert "configuration" in res.answer.lower()
    assert len(res.citations) > 0
    pages = [c.pdf_page for c in res.citations]
    assert 8 in pages or 25 in pages


def test_q2_ordinary_application(pipeline):
    """Q2: Verify definition of ordinary application."""
    q = "What is an ordinary application?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)

    assert res.is_answered is True
    assert "does not claim priority" in res.answer.lower()
    pages = [c.pdf_page for c in res.citations]
    assert 11 in pages


def test_q3_reciprocity_application(pipeline):
    """Q3: Verify reciprocity application and 6-month filing period."""
    q = "What is a reciprocity application, and what filing period does the manual describe?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)

    assert res.is_answered is True
    assert "six months" in res.answer.lower()
    assert "priority" in res.answer.lower()
    pages = [c.pdf_page for c in res.citations]
    assert 11 in pages or 9 in pages or 12 in pages or 16 in pages


def test_q4_address_destination(pipeline):
    """Q4: Verify Controller of Designs address."""
    q = "To whom and at what address should a design registration application be addressed according to the manual?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)

    assert res.is_answered is True
    assert "Controller of Designs" in res.answer
    assert "Kolkata" in res.answer
    assert "700091" in res.answer
    pages = [c.pdf_page for c in res.citations]
    assert 11 in pages


def test_q5_patent_office_locations(pipeline):
    """Q5: Verify four Patent Office locations."""
    q = "Which four Patent Office locations are mentioned in the manual?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)

    assert res.is_answered is True
    for city in ["Delhi", "Mumbai", "Kolkata", "Chennai"]:
        assert city.lower() in res.answer.lower()
    pages = [c.pdf_page for c in res.citations]
    assert 9 in pages or 23 in pages


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Live test requires GOOGLE_API_KEY")
def test_live_gemini_integration(pipeline):
    """Live integration test using Google Gemini chat model if key is present."""
    q = "What is an ordinary application?"
    res = pipeline.run(q)  # uses live LLM

    assert res.is_answered is True
    assert "priority" in res.answer.lower()
    assert len(res.citations) > 0
    assert any(c.pdf_page == 11 for c in res.citations)


def test_assignment_q1_copyright_duration(pipeline):
    """Assignment Q1: Duration of copyright in registered design and extension."""
    q = "How long does a proprietor have copyright in a registered design, and can it be extended?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)
    assert res.is_answered is True
    assert "ten years" in res.answer.lower()
    assert "five years" in res.answer.lower()
    assert any(c.pdf_page in [36, 39] for c in res.citations)


def test_assignment_q2_classification_system(pipeline):
    """Assignment Q2: Classification system for registered designs."""
    q = "Which classification system is used to register designs?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)
    assert res.is_answered is True
    assert "locarno" in res.answer.lower()
    assert any(c.pdf_page == 20 for c in res.citations)


def test_assignment_q3_trademark_fee(pipeline):
    """Assignment Q3: Government fee to register a trademark (must be Not found)."""
    q = "What is the government fee to register a trademark?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)
    assert res.is_answered is False
    assert res.answer == "Not found"
    assert len(res.citations) == 0


def test_assignment_q4_current_controller_general(pipeline):
    """Assignment Q4: Current Controller General of Patents (must be Not found)."""
    q = "Who is the current Controller General of Patents?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)
    assert res.is_answered is False
    assert res.answer == "Not found"
    assert len(res.citations) == 0


def test_assignment_q5_design_registration_fee(pipeline):
    """Assignment Q5: Fee to register a design (manual does not provide rupee fees; must be Not found)."""
    q = "What is the fee to register a design?"
    res = pipeline.run(q, mock_llm=deterministic_mock_llm)
    assert res.is_answered is False
    assert res.answer == "Not found"
    assert len(res.citations) == 0

