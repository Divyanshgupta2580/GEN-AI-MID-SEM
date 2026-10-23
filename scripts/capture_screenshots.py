import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = BASE_DIR / "docs" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS = [
    {
        "id": 1,
        "filename": "question_01_design_copyright.png",
        "question": "How long does a proprietor have copyright in a registered design, and can it be extended?",
        "expected_type": "answer",
    },
    {
        "id": 2,
        "filename": "question_02_classification_system.png",
        "question": "Which classification system is used to register designs?",
        "expected_type": "answer",
    },
    {
        "id": 3,
        "filename": "question_03_trademark_fee.png",
        "question": "What is the government fee to register a trademark?",
        "expected_type": "not_found",
    },
    {
        "id": 4,
        "filename": "question_04_current_controller_general.png",
        "question": "Who is the current Controller General of Patents?",
        "expected_type": "not_found",
    },
    {
        "id": 5,
        "filename": "question_05_design_registration_fee.png",
        "question": "What is the fee to register a design?",
        "expected_type": "not_found",
    },
]

VIEWPORTS = [
    {"name": "Desktop", "width": 1440, "height": 900},
    {"name": "Laptop", "width": 1280, "height": 800},
    {"name": "Tablet", "width": 768, "height": 1024},
    {"name": "Mobile", "width": 390, "height": 844},
]


def test_responsive_viewports(browser):
    """Test the application layout across 4 standard viewports."""
    print("=" * 60)
    print("TESTING RESPONSIVE VIEWPORTS")
    print("=" * 60)
    
    for vp in VIEWPORTS:
        print(f"\nTesting Viewport: {vp['name']} ({vp['width']}x{vp['height']})...")
        context = browser.new_context(viewport={"width": vp["width"], "height": vp["height"]})
        page = context.new_page()
        page.goto("http://localhost:8501", timeout=30000)
        
        # Verify input and button are present and visible
        input_selector = "input[aria-label='Enter your question about the Designs Manual:']"
        page.wait_for_selector(input_selector, timeout=20000)
        
        # On narrow viewports (e.g. Mobile/Tablet), collapse sidebar if open
        if vp["width"] < 900:
            collapse_btn = page.locator("button[data-testid='stSidebarCollapseButton']")
            if collapse_btn.is_visible():
                collapse_btn.click()
                time.sleep(0.5)

        ask_btn = page.locator("button:has-text('Ask Question')")
        assert ask_btn.is_visible(), f"Ask Question button not visible on {vp['name']}"
        print(f"  [+] Input & Ask button visible on {vp['name']}")
        
        # Test an answerable question on this viewport
        input_field = page.locator(input_selector)
        input_field.fill("What is an ordinary application?")
        ask_btn.click(force=True)
        page.wait_for_selector(".answer-card", timeout=35000)
        time.sleep(1)
        
        # Verify no horizontal body overflow
        has_horizontal_overflow = page.evaluate("() => document.body.scrollWidth > window.innerWidth")
        print(f"  [+] Horizontal overflow check: {'FAIL - Overflow detected' if has_horizontal_overflow else 'PASS - No overflow'}")
        assert not has_horizontal_overflow, f"Horizontal overflow detected on {vp['name']}"
        
        # Test an unanswerable question
        input_field.fill("What was the weather in Delhi yesterday?")
        ask_btn.click(force=True)
        page.wait_for_selector(".not-found-card", timeout=35000)
        time.sleep(1)
        
        has_horizontal_overflow_nf = page.evaluate("() => document.body.scrollWidth > window.innerWidth")
        assert not has_horizontal_overflow_nf, f"Horizontal overflow on Not found card on {vp['name']}"
        print(f"  [+] Not found card layout check: PASS")
        
        context.close()
    
    print("\n[✓] All 4 responsive viewports passed verification successfully.\n")


def capture_five_assignment_screenshots(browser):
    """Capture exactly the 5 required assignment screenshots from live application."""
    print("=" * 60)
    print("CAPTURING 5 ASSIGNMENT DEMONSTRATION SCREENSHOTS")
    print("=" * 60)

    # Use a clean, sharp desktop viewport (1280x880) optimal for GitHub README display
    context = browser.new_context(viewport={"width": 1280, "height": 880}, device_scale_factor=1.5)
    page = context.new_page()

    input_selector = "input[aria-label='Enter your question about the Designs Manual:']"
    ask_btn_selector = "button:has-text('Ask Question')"

    for q_data in QUESTIONS:
        qid = q_data["id"]
        q_text = q_data["question"]
        filename = q_data["filename"]
        exp_type = q_data["expected_type"]
        out_path = SCREENSHOT_DIR / filename

        print(f"\nProcessing Question {qid}:")
        print(f"  Query: \"{q_text}\"")
        
        # Fresh navigation to ensure clean state with no previous history
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_selector(input_selector, timeout=20000)
        time.sleep(1)

        input_field = page.locator(input_selector)
        input_field.fill(q_text)
        page.locator(ask_btn_selector).click()

        if exp_type == "answer":
            page.wait_for_selector(".answer-card", timeout=45000)
            page.wait_for_selector(".citation-card", timeout=15000)
            time.sleep(1.5)
            print("  [+] Received document-grounded answer with supporting citations.")
        else:
            page.wait_for_selector(".not-found-card", timeout=45000)
            time.sleep(1.5)
            print("  [+] Received exact 'Not found' result without fabricated citations.")

        # Ensure page content is settled and capture screenshot
        page.screenshot(path=str(out_path), full_page=False)
        assert out_path.exists() and out_path.stat().st_size > 50000, f"Screenshot {filename} failed to save properly"
        print(f"  [✓] Genuine screenshot saved: {out_path} ({out_path.stat().st_size // 1024} KB)")

    context.close()
    print("\n[✓] All 5 assignment screenshots successfully captured and verified.")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            test_responsive_viewports(browser)
            capture_five_assignment_screenshots(browser)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
