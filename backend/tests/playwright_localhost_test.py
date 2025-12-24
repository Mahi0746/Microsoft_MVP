from playwright.sync_api import sync_playwright

URL='http://localhost:3000'

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    c = b.new_context()
    page = c.new_page()
    print('navigating to', URL)
    page.goto(URL, timeout=10000)
    print('loaded')
    print('title:', page.title())
    b.close()
