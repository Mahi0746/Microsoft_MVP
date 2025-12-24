from playwright.sync_api import sync_playwright
import time

FRONTEND = 'http://127.0.0.1:3000'
BACKEND = 'http://127.0.0.1:8000'
EMAIL = 'testuser@healthsync.com'
PASSWORD = 'TestPassword123!'


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.goto(FRONTEND + '/auth/login')
        time.sleep(1)

        # Fill login form
        # Try common selectors
        try:
            page.fill('input[name="email"]', EMAIL)
            page.fill('input[name="password"]', PASSWORD)
        except Exception:
            # Try placeholders
            page.fill('input[placeholder="Email"]', EMAIL)
            page.fill('input[placeholder="Password"]', PASSWORD)

        # Click submit
        try:
            page.click('button[type="submit"]')
        except Exception:
            page.press('input[name="password"]', 'Enter')

        # Wait for navigation or token in localStorage
        page.wait_for_timeout(2000)

        # Check localStorage for token
        token = page.evaluate("() => window.localStorage.getItem('token')")
        print('token_present', bool(token))

        # Visit a protected page
        page.goto(FRONTEND + '/dashboard')
        page.wait_for_timeout(1000)
        html = page.content()
        print('dashboard_title_snippet', html[:200])

        browser.close()


if __name__ == '__main__':
    run()
