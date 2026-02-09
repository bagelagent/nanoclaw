---
name: playwright
description: Advanced browser automation with Playwright. Handles cookies, anti-bot protection, JavaScript execution, and complex web interactions. Use when agent-browser isn't sufficient.
---

# Playwright Browser Automation Skill

Advanced browser automation using Playwright for complex web interactions that require:
- Cookie injection before page load
- Anti-bot/Cloudflare bypass
- JavaScript execution
- Complex form interactions
- Session persistence

## When to Use This vs agent-browser

**Use Playwright when:**
- Need to load cookies before navigation (DoorDash, authenticated sites)
- Site has aggressive anti-bot protection (Cloudflare)
- Need to execute custom JavaScript in page context
- Require precise timing control and wait strategies
- Need to handle complex authentication flows

**Use agent-browser when:**
- Simple browsing and clicking
- No authentication required
- No anti-bot protection
- Quick information extraction

## Installation

Playwright is installed in the container at `/workspace/project/node_modules/playwright`.

## Basic Usage

```javascript
const playwright = require('playwright');
const fs = require('fs');

// Launch browser
const browser = await playwright.chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox']
});

// Create context with cookies
const cookies = JSON.parse(fs.readFileSync('/path/to/cookies.json', 'utf8'));
const context = await browser.newContext({
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
});
await context.addCookies(cookies);

// Navigate with cookies already set
const page = await context.newPage();
await page.goto('https://example.com');

// Interact with page
await page.click('button#submit');
await page.fill('input[name="search"]', 'query');

// Wait for elements
await page.waitForSelector('.results');

// Extract data
const data = await page.evaluate(() => {
  return document.querySelector('.data').textContent;
});

// Cleanup
await browser.close();
```

## Cookie Management

### Loading Cookies from File

```javascript
// Cookies should be in Playwright format:
[
  {
    "name": "cookie_name",
    "value": "cookie_value",
    "domain": ".example.com",
    "path": "/",
    "expires": 1234567890, // Unix timestamp
    "httpOnly": true,
    "secure": true,
    "sameSite": "Lax"
  }
]

// Load before navigation
await context.addCookies(cookies);
```

### Saving Cookies

```javascript
const cookies = await context.cookies();
fs.writeFileSync('/path/to/cookies.json', JSON.stringify(cookies, null, 2));
```

## Anti-Bot Strategies

### Bypass Cloudflare

1. **Use realistic browser context:**
```javascript
const context = await browser.newContext({
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
  viewport: { width: 1920, height: 1080 },
  locale: 'en-US',
  timezoneId: 'America/Los_Angeles',
});
```

2. **Load cookies before navigation** (most important):
```javascript
await context.addCookies(cookies); // BEFORE goto()
await page.goto(url);
```

3. **Add delays between actions:**
```javascript
await page.waitForTimeout(1000); // Wait 1 second
```

4. **Handle waiting room challenges:**
```javascript
// Wait for challenge to complete
await page.waitForSelector('.main-content', { timeout: 30000 });
```

## Common Patterns

### Form Filling

```javascript
await page.fill('input[name="email"]', 'user@example.com');
await page.fill('input[name="password"]', 'password');
await page.click('button[type="submit"]');
await page.waitForNavigation();
```

### Waiting for Dynamic Content

```javascript
// Wait for element
await page.waitForSelector('.loaded-content');

// Wait for network idle
await page.waitForLoadState('networkidle');

// Wait for specific condition
await page.waitForFunction(() => {
  return document.querySelectorAll('.item').length > 0;
});
```

### Taking Screenshots

```javascript
// Full page
await page.screenshot({ path: '/tmp/screenshot.png', fullPage: true });

// Specific element
const element = await page.$('.target');
await element.screenshot({ path: '/tmp/element.png' });
```

### JavaScript Execution

```javascript
// Execute in page context
const result = await page.evaluate(() => {
  return {
    title: document.title,
    items: Array.from(document.querySelectorAll('.item')).map(el => el.textContent)
  };
});
```

## Helper Script Template

Create reusable scripts in `/workspace/group/scripts/`:

```bash
#!/bin/bash
# Example: playwright-browser.js

node << 'EOF'
const playwright = require('playwright');
const fs = require('fs');

(async () => {
  const browser = await playwright.chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const context = await browser.newContext({
      userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    });

    // Load cookies if available
    if (fs.existsSync('/workspace/group/.cookies.json')) {
      const cookies = JSON.parse(fs.readFileSync('/workspace/group/.cookies.json', 'utf8'));
      await context.addCookies(cookies);
    }

    const page = await context.newPage();

    // Your automation here
    await page.goto('https://example.com');

    // Take screenshot
    await page.screenshot({ path: '/workspace/group/tmp/result.png' });

    console.log('Success!');
  } finally {
    await browser.close();
  }
})();
EOF
```

## Error Handling

```javascript
try {
  await page.goto(url, { timeout: 30000 });
} catch (error) {
  if (error.message.includes('timeout')) {
    console.error('Page load timeout');
  } else if (error.message.includes('net::ERR_')) {
    console.error('Network error');
  } else {
    console.error('Unknown error:', error.message);
  }
}
```

## Best Practices

1. **Always close browsers** - Use try/finally blocks
2. **Handle timeouts gracefully** - Set appropriate timeout values
3. **Wait for elements** - Don't assume instant page loads
4. **Use realistic user agents** - Match real browsers
5. **Add human-like delays** - Don't automate too fast
6. **Save cookies** - Maintain session state between runs
7. **Take screenshots** - Debug issues visually
8. **Log actions** - Track what the automation is doing

## Security Notes

- Cookies are sensitive credentials - never log them
- Store cookies in `/workspace/group/.{service}-cookies.json`
- Set appropriate file permissions (600)
- Rotate cookies periodically
- Never commit cookies to version control

## Troubleshooting

**Browser won't launch:**
- Check if Playwright browsers are installed: `npx playwright install chromium`
- Verify container has necessary dependencies

**Cloudflare challenges:**
- Ensure cookies are loaded BEFORE navigation
- Use realistic user agent and viewport
- Add delays between actions
- Check cookie expiration dates

**Elements not found:**
- Use longer timeouts: `waitForSelector('.element', { timeout: 60000 })`
- Check if content loads dynamically
- Take screenshot to debug: `await page.screenshot()`

**Memory issues:**
- Always close browsers in finally blocks
- Don't keep multiple browsers open
- Clear context between operations if needed
