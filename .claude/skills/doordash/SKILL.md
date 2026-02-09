---
name: doordash
description: Order food from DoorDash using authenticated browser automation. Search restaurants, browse menus, add items to cart, and place orders.
---

# DoorDash Ordering Skill

Automated food ordering through DoorDash using browser automation with saved authentication cookies.

## Prerequisites

**Authentication Cookies Required:**
- Stored in `/workspace/group/.doordash-cookies.json`
- User will provide a screenshot of their DoorDash cookies from browser DevTools
- Extract and save cookies for automated authentication

## Cookie Setup Process

When user provides cookie screenshot:
1. Extract all cookies from the screenshot (domain: doordash.com)
2. Save to `/workspace/group/.doordash-cookies.json` in format:
```json
[
  {
    "name": "cookie_name",
    "value": "cookie_value",
    "domain": ".doordash.com",
    "path": "/",
    "secure": true,
    "httpOnly": true
  }
]
```

## Ordering Process

### 1. Initialize Browser with Cookies
```bash
# Load cookies and navigate to DoorDash
agent-browser open "https://www.doordash.com"
# Inject cookies using browser DevTools if needed
```

### 2. Search for Restaurants
```bash
# User might say: "order from Chipotle" or "find Thai food nearby"
agent-browser type [search-box-ref] "restaurant name or cuisine"
agent-browser click [search-button-ref]
```

### 3. Browse Menu
```bash
# Navigate to restaurant page
agent-browser click [restaurant-ref]
# Take snapshot to see menu items
agent-browser snapshot -i
```

### 4. Add Items to Cart
```bash
# For each item user wants:
agent-browser click [menu-item-ref]
# Handle customizations if modal appears
agent-browser click [customization-refs]
agent-browser click [add-to-cart-ref]
```

### 5. Review Cart & Checkout
```bash
# Go to cart
agent-browser click [cart-icon-ref]
# Review order
agent-browser snapshot
# Proceed to checkout
agent-browser click [checkout-ref]
```

### 6. Place Order
```bash
# Confirm delivery address
# Select payment method (should be pre-saved)
# Review order total
# Place order
agent-browser click [place-order-ref]
```

## User Interaction Patterns

### Simple Orders
User: "Order a burrito bowl from Chipotle"
1. Search for Chipotle
2. Find burrito bowl on menu
3. Ask user for customizations (or use defaults)
4. Add to cart and checkout

### Complex Orders
User: "Order dinner for 3 from Thai restaurant, I want pad thai, get something for the kids too"
1. Search Thai restaurants nearby
2. Present options with ratings
3. Ask user to choose restaurant
4. Browse menu and suggest kid-friendly items
5. Build cart with multiple items
6. Confirm before placing order

### Reorders
User: "Reorder my usual from Chipotle"
1. Check order history
2. Find last Chipotle order
3. Add same items to cart
4. Confirm and place order

## Important Notes

### Always Confirm Before Placing
- Show order summary with items and total
- Get explicit user confirmation before clicking "Place Order"
- Never place orders without confirmation

### Handle Edge Cases
- Restaurant closed → suggest alternatives
- Items unavailable → ask for substitutions
- Minimum order not met → suggest additional items
- Payment issues → alert user to update payment method

### Store Preferences
Remember in CLAUDE.md:
- Favorite restaurants
- Common orders
- Dietary restrictions
- Default delivery address
- Tip preferences

## Security

**Cookie Safety:**
- Cookies stored locally in `/workspace/group/.doordash-cookies.json`
- Never log cookie values
- Cookies are sensitive - treat as credentials
- Rotate cookies periodically (user will update)

## Example Session

```
User: "Order pad thai from my favorite Thai place"
Agent: "I'll order pad thai from Thai House (your usual). Let me check the menu."
[Opens DoorDash, navigates to Thai House, finds pad thai]
Agent: "Found Pad Thai ($14.99). Any customizations? (spice level, protein, extras)"

User: "Medium spicy, add chicken"

Agent: "Got it. Adding:
- Pad Thai with chicken, medium spicy: $14.99
- Subtotal: $14.99
- Delivery fee: $2.99
- Service fee: $2.50
- Tax: $1.35
- Tip (18%): $2.70
- Total: $24.53

Ready to place this order?"

User: "Yes"

Agent: [Places order]
"Order placed! ETA: 35-45 minutes. Order #12345"
```

## Testing

Before going live, test the browser automation:
1. Open DoorDash with cookies
2. Verify authentication works (logged in as user)
3. Navigate through menu without placing order
4. Confirm all steps work smoothly

## Future Enhancements

- **Voice ordering**: "Hey, order my usual from Chipotle"
- **Schedule orders**: "Order pizza for delivery at 6pm"
- **Budget tracking**: Track monthly food delivery spending
- **Favorites**: Quick access to frequently ordered items
- **Group orders**: Order for family with everyone's preferences
