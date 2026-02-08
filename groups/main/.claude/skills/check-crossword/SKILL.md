---
name: check-crossword
description: Find errors in crossword puzzles by comparing user's grid to the solution from Rex Parker's blog. Use when user needs help identifying incorrect squares.
---

# Check Crossword Skill

You are helping the user find errors in their crossword puzzle by comparing it to the solution.

## Process

1. **Get the user's grid image** - they will provide a screenshot
2. **Find the solution** - Use agent-browser to visit Rex Parker's blog at https://rexwordpuzzle.blogspot.com
3. **Take screenshots** - Capture the solution grid from Rex Parker's post
4. **Systematic comparison**:
   - Read both images carefully
   - Compare row by row, column by column
   - Look for visual differences in letter placement
   - Pay special attention to areas where letters look similar (like A vs E, O vs Q, etc.)
5. **Identify specific incorrect squares**:
   - Note the exact position (row/column or crossing clues)
   - Identify which answers are affected
   - Explain what the user has vs what it should be

## Important Guidelines

- **Be methodical**: Compare grids systematically, don't jump to conclusions
- **Verify letter counts**: If an answer doesn't match the expected length, you've made an error
- **Double-check**: Look at the user's grid multiple times before declaring what's wrong
- **Be specific**: Say exactly which intersection is wrong (e.g., "TINMAN/KABABS cross")
- **Admit if unclear**: If the images are hard to read, ask the user for clarification

## Common Pitfalls to Avoid

1. Don't guess at answers without carefully reading both grids
2. Don't suggest impossible answers (wrong letter counts)
3. Don't compare just a portion - check the entire grid systematically
4. Don't assume - verify each letter carefully

## Output Format

When you find the error(s), report:
- The incorrect word(s) the user has
- The correct word(s) it should be
- The specific square(s) that are wrong (with the intersection if applicable)
- What letter the user has vs. what it should be
