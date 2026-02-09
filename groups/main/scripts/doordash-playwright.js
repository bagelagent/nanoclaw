#!/usr/bin/env node
/**
 * DoorDash automation using Playwright
 *
 * Usage:
 *   node doordash-playwright.js <command> [...args]
 *
 * Commands:
 *   orders                    - Get recent orders
 *   search <query>           - Search for restaurants
 *   menu <restaurant_url>    - Get restaurant menu
 *   screenshot <url>         - Take screenshot of page
 */

const playwright = require('/workspace/project/node_modules/playwright');
const fs = require('fs');
const path = require('path');

const COOKIES_PATH = '/workspace/group/.doordash-cookies.json';
const TMP_DIR = '/workspace/group/tmp';

// Ensure tmp directory exists
if (!fs.existsSync(TMP_DIR)) {
  fs.mkdirSync(TMP_DIR, { recursive: true });
}

async function createBrowser() {
  const browser = await playwright.chromium.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-blink-features=AutomationControlled'
    ]
  });

  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    viewport: { width: 1920, height: 1080 },
    locale: 'en-US',
    timezoneId: 'America/Los_Angeles',
  });

  // Load cookies if available
  if (fs.existsSync(COOKIES_PATH)) {
    const cookies = JSON.parse(fs.readFileSync(COOKIES_PATH, 'utf8'));
    await context.addCookies(cookies);
    console.log(`Loaded ${cookies.length} cookies`);
  } else {
    console.warn('No cookies found at', COOKIES_PATH);
  }

  return { browser, context };
}

async function getOrders() {
  const { browser, context } = await createBrowser();

  try {
    const page = await context.newPage();
    console.log('Navigating to DoorDash orders page...');

    await page.goto('https://www.doordash.com/orders/', {
      waitUntil: 'domcontentloaded',
      timeout: 30000
    });

    // Wait a bit for any challenges
    await page.waitForTimeout(3000);

    // Take screenshot for debugging
    const screenshotPath = path.join(TMP_DIR, 'doordash-orders.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log('Screenshot saved:', screenshotPath);

    // Try to extract order data
    const orderData = await page.evaluate(() => {
      // Check if we're logged in
      const isLoggedIn = document.body.textContent.includes('Orders') ||
                        document.querySelector('[data-testid="order"]');

      if (!isLoggedIn) {
        return { error: 'Not logged in or Cloudflare challenge present' };
      }

      // Try to find orders
      const orders = [];
      const orderElements = document.querySelectorAll('[data-testid="order"], .order-card, [class*="OrderCard"]');

      orderElements.forEach(el => {
        const text = el.textContent;
        orders.push({
          text: text.substring(0, 200), // First 200 chars
          html: el.outerHTML.substring(0, 300)
        });
      });

      return {
        url: window.location.href,
        title: document.title,
        ordersFound: orders.length,
        orders: orders.slice(0, 3), // First 3 orders
        bodyPreview: document.body.textContent.substring(0, 500)
      };
    });

    console.log(JSON.stringify(orderData, null, 2));
    return orderData;

  } finally {
    await browser.close();
  }
}

async function searchRestaurants(query) {
  const { browser, context } = await createBrowser();

  try {
    const page = await context.newPage();
    console.log(`Searching for: ${query}`);

    await page.goto('https://www.doordash.com', {
      waitUntil: 'domcontentloaded',
      timeout: 30000
    });

    await page.waitForTimeout(2000);

    // Try to find and use search box
    const searchInput = await page.$('input[placeholder*="Search"], input[type="search"]');
    if (searchInput) {
      await searchInput.fill(query);
      await page.waitForTimeout(1000);
      await page.keyboard.press('Enter');
      await page.waitForTimeout(3000);
    }

    const screenshotPath = path.join(TMP_DIR, 'doordash-search.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log('Screenshot saved:', screenshotPath);

    const results = await page.evaluate(() => {
      return {
        url: window.location.href,
        title: document.title,
        bodyPreview: document.body.textContent.substring(0, 500)
      };
    });

    console.log(JSON.stringify(results, null, 2));
    return results;

  } finally {
    await browser.close();
  }
}

async function takeScreenshot(url) {
  const { browser, context } = await createBrowser();

  try {
    const page = await context.newPage();
    console.log(`Taking screenshot of: ${url}`);

    await page.goto(url, {
      waitUntil: 'domcontentloaded',
      timeout: 30000
    });

    await page.waitForTimeout(3000);

    const screenshotPath = path.join(TMP_DIR, 'doordash-page.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log('Screenshot saved:', screenshotPath);

    return screenshotPath;

  } finally {
    await browser.close();
  }
}

// Main CLI handler
async function main() {
  const command = process.argv[2];
  const args = process.argv.slice(3);

  try {
    switch (command) {
      case 'orders':
        await getOrders();
        break;

      case 'search':
        if (!args[0]) {
          console.error('Usage: search <query>');
          process.exit(1);
        }
        await searchRestaurants(args[0]);
        break;

      case 'screenshot':
        if (!args[0]) {
          console.error('Usage: screenshot <url>');
          process.exit(1);
        }
        await takeScreenshot(args[0]);
        break;

      default:
        console.error('Unknown command:', command);
        console.error('Available commands: orders, search, screenshot');
        process.exit(1);
    }
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { createBrowser, getOrders, searchRestaurants, takeScreenshot };
