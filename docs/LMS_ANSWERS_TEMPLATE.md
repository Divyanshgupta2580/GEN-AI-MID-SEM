# LMS Submission Answers Template

> [!IMPORTANT]
> The five official questions from the academic LMS portal have not been pre-loaded into this project specification.
> **Do not guess or fabricate external LMS questions.**
>
> When the official LMS questions are assigned, execute them against the running application:
> ```bash
> # Run via interactive Streamlit UI
> streamlit run app.py
> 
> # Or execute via Python pipeline directly
> python -c '
> from src.rag_app.rag_pipeline import RAGPipeline
> from src.rag_app.config import config
> p = RAGPipeline(config)
> print(p.run("<ENTER_LMS_QUESTION_HERE>").formatted_display())
> '
> ```
> Transfer the exact generated answers and programmatic citations into the five slots below for submission.

---

### LMS Question 1
- **Assigned Question:** [Enter LMS Question 1 here]
- **Document-Grounded Answer:** [Transfer generated answer here]
- **PDF Page Reference:** [e.g., PDF page X, Section Y]
- **Supporting Document Excerpt:** [Paste supporting excerpt from retrieved evidence]

---

### LMS Question 2
- **Assigned Question:** [Enter LMS Question 2 here]
- **Document-Grounded Answer:** [Transfer generated answer here]
- **PDF Page Reference:** [e.g., PDF page X, Section Y]
- **Supporting Document Excerpt:** [Paste supporting excerpt from retrieved evidence]

---

### LMS Question 3
- **Assigned Question:** [Enter LMS Question 3 here]
- **Document-Grounded Answer:** [Transfer generated answer here]
- **PDF Page Reference:** [e.g., PDF page X, Section Y]
- **Supporting Document Excerpt:** [Paste supporting excerpt from retrieved evidence]

---

### LMS Question 4
- **Assigned Question:** [Enter LMS Question 4 here]
- **Document-Grounded Answer:** [Transfer generated answer here]
- **PDF Page Reference:** [e.g., PDF page X, Section Y]
- **Supporting Document Excerpt:** [Paste supporting excerpt from retrieved evidence]

---

### LMS Question 5
- **Assigned Question:** [Enter LMS Question 5 here]
- **Document-Grounded Answer:** [Transfer generated answer here]
- **PDF Page Reference:** [e.g., PDF page X, Section Y]
- **Supporting Document Excerpt:** [Paste supporting excerpt from retrieved evidence]

---

## Submission Checklist for Student
- [ ] Entered all 5 questions verbatim from the LMS assignment prompt.
- [ ] Verified that each answer is grounded strictly in the official *Manual of Designs Practice & Procedure*.
- [ ] Verified physical PDF page citations match the 87-page document.
- [ ] Confirmed that any unsupported fact or question is answered strictly with `Not found`.
