from playwright.sync_api import sync_playwright
import requests, time, sys

FRONTEND_URLS = ['http://127.0.0.1:3000', 'http://localhost:3000']
BACKEND = 'http://127.0.0.1:8000'
EMAIL = 'testuser@healthsync.com'
PASSWORD = 'TestPassword123!'

# Wait until frontend responds
ready = False
for url in FRONTEND_URLS:
    for i in range(20):
        try:
            r = requests.get(url, timeout=2)
            print(f'Frontend {url} responded', r.status_code)
            FRONTEND = url
            ready = True
            break
        except Exception as e:
            print(f'Waiting for {url}... attempt', i, 'error:', e)
            time.sleep(1)
    if ready:
        break

if not ready:
    print('Frontend not reachable on any URL, aborting')
    sys.exit(2)

with sync_playwright() as p:
    try:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_navigation_timeout(60000)

        login_path = FRONTEND + '/auth/login'
        print('Navigating to', login_path)
        page.goto(login_path)
        time.sleep(1)

        # Try filling common fields
        try:
            if page.locator('input[name="email"]').count() > 0:
                page.fill('input[name="email"]', EMAIL)
            elif page.locator('input[placeholder="Email"]').count() > 0:
                page.fill('input[placeholder="Email"]', EMAIL)
        except Exception as e:
            print('Could not fill email:', e)

        try:
            if page.locator('input[name="password"]').count() > 0:
                page.fill('input[name="password"]', PASSWORD)
            elif page.locator('input[placeholder="Password"]').count() > 0:
                page.fill('input[placeholder="Password"]', PASSWORD)
        except Exception as e:
            print('Could not fill password:', e)

        # Submit
        try:
            if page.locator('button[type="submit"]').count() > 0:
                page.click('button[type="submit"]')
            else:
                page.press('input[name="password"]', 'Enter')
        except Exception as e:
            print('Could not submit form:', e)

        page.wait_for_timeout(3000)
        token = page.evaluate("() => window.localStorage.getItem('token')")
        print('token_present', bool(token))

        page.goto(FRONTEND + '/dashboard')
        page.wait_for_timeout(2000)
        print('dashboard snippet:', page.content()[:300])

        browser.close()
    except Exception as e:
        print('Playwright error:', e)
        sys.exit(3)

print('Playwright test completed')
