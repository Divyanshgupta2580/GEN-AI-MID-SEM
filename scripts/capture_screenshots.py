import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = BASE_DIR / "docs" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def capture_all():
    print("Launching Chromium for genuine screenshot capture...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_selector("input[aria-label='Enter your question about the Designs Manual:']", timeout=20000)
        time.sleep(2)

        # -------------------------------------------------------------
        # Screenshot 1: Answerable Question (What is an ordinary application?)
        # -------------------------------------------------------------
        print("Capturing Screenshot 1: 'What is an ordinary application?'...")
        input_field = page.locator("input[aria-label='Enter your question about the Designs Manual:']")
        input_field.fill("What is an ordinary application?")
        
        # Click Ask Question button
        page.locator("button:has-text('Ask Question')").click()
        # Wait for answer card to appear
        page.wait_for_selector(".answer-card", timeout=30000)
        time.sleep(2)

        p1 = SCREENSHOT_DIR / "01_answer_ordinary_application.png"
        page.screenshot(path=str(p1), full_page=False)
        print(f"[+] Saved Screenshot 1 to {p1}")

        # -------------------------------------------------------------
        # Screenshot 2: Second Answerable Question (Address and Controller)
        # -------------------------------------------------------------
        print("Capturing Screenshot 2: 'To whom and at what address should a design registration application be addressed according to the manual?'...")
        input_field.fill("To whom and at what address should a design registration application be addressed according to the manual?")
        page.locator("button:has-text('Ask Question')").click()
        page.wait_for_selector(".answer-card", timeout=30000)
        time.sleep(2)

        p2 = SCREENSHOT_DIR / "02_answer_address_citations.png"
        page.screenshot(path=str(p2), full_page=False)
        print(f"[+] Saved Screenshot 2 to {p2}")

        # -------------------------------------------------------------
        # Screenshot 3: Unanswerable Question (Weather in Delhi) -> Not found
        # -------------------------------------------------------------
        print("Capturing Screenshot 3: 'What was the weather in Delhi yesterday?'...")
        input_field.fill("What was the weather in Delhi yesterday?")
        page.locator("button:has-text('Ask Question')").click()
        page.wait_for_selector(".not-found-card", timeout=30000)
        time.sleep(2)

        p3 = SCREENSHOT_DIR / "03_unanswerable_not_found.png"
        page.screenshot(path=str(p3), full_page=False)
        print(f"[+] Saved Screenshot 3 to {p3}")

        browser.close()
        print("All genuine screenshots captured successfully.")


if __name__ == "__main__":
    capture_all()
