---
name: ride-prices
description: Compare Uber, Lyft, and other rideshare prices between two locations using RideGuru. Use when user asks about ride prices, fare comparison, or which rideshare is cheaper.
---

# Ride Price Comparison Skill

Compare Uber, Lyft, and other rideshare prices between two locations using RideGuru.

## When to Use

- User asks "how much is an Uber/Lyft from X to Y?"
- User asks to compare ride prices
- User asks "which is cheaper, Uber or Lyft?"
- User asks about ride/fare estimates

## Process

1. **Extract pickup and destination** from the user's message. If unclear, ask for clarification.

2. **Open RideGuru** using agent-browser:
   ```bash
   agent-browser open "https://ride.guru"
   ```

3. **Handle Cloudflare** — RideGuru has a Cloudflare check. Wait a moment and look for the checkbox:
   ```bash
   agent-browser snapshot -i
   ```
   If you see a "Verify you are human" checkbox, click it and wait 5-8 seconds:
   ```bash
   agent-browser click <checkbox-ref>
   ```

4. **Fill in the pickup location**:
   ```bash
   agent-browser fill <start-textbox-ref> "pickup address"
   ```
   Wait 2 seconds for autocomplete, then click the best matching suggestion from the dropdown list.

5. **Fill in the destination**:
   ```bash
   agent-browser fill <end-textbox-ref> "destination address"
   ```
   Wait 2 seconds for autocomplete, then click the best matching suggestion from the dropdown list.

6. **Click "GET ESTIMATES"** button.

7. **Wait 3-5 seconds** for results to load, then **click the "Sedan" tab** to see standard ride prices.

8. **Take a screenshot** to read the price chart:
   ```bash
   agent-browser screenshot /workspace/group/tmp/ride-prices.png
   ```

9. **Read the screenshot** to extract all prices from the bar chart on the left side and the cards.

10. **Close the browser**:
    ```bash
    agent-browser close
    ```

11. **Report results** via send_message, formatted like:
    ```
    Ride prices from [pickup] to [destination]:

    *Cheapest options:*
    • Uber Green: $XX
    • Uber X: $XX
    • Lyft: $XX

    *Premium options:*
    • Uber Comfort: $XX
    • Uber XL: $XX
    • Lyft XL: $XX

    *Luxury:*
    • Uber Black: $XX
    • Lyft Lux Black: $XX

    Winner: [cheapest service] at $XX
    ```

## Important Notes

- The bar chart on the left side of the results page shows all prices sorted cheapest to most expensive
- Prices marked with ⚠️ are estimates and may vary
- RideGuru estimates may not match real-time surge pricing
- If RideGuru is down or Cloudflare blocks you, tell the user and suggest they check the Uber/Lyft apps directly
- Send the screenshot to the user along with the text summary so they can see the full comparison
- If the user provides shorthand locations (e.g., "SFO", "home"), interpret them reasonably. For "home", check CLAUDE.md for the user's address if available.
