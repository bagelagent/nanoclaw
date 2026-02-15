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
4. **Read both grids completely BEFORE analyzing** - Don't start comparing until you can see both full grids clearly
5. **Identify blue/highlighted squares** - These show exactly where errors are located
6. **For EACH blue square, trace the complete crossing words**:
   - Read the full ACROSS answer character-by-character from user's grid
   - Read the full DOWN answer character-by-character from user's grid
   - Count the letters to verify length
   - Do the same for the solution grid
7. **Character-by-character comparison** - Write out both versions:
   ```
   User has:    S-A-V-S (4 letters)
   Solution:    S-A-S-S (4 letters)
   Difference:  Position 3 is V vs S
   ```
8. **Verify letter counts match** - If lengths don't match between user and solution, you made a reading error - re-read carefully
9. **Report the specific crossing answers** - State exactly which ACROSS and DOWN answers intersect at the error

## Important Guidelines

- **SLOW DOWN**: The biggest source of errors is rushing. Take time to read carefully.
- **NO GUESSING**: Never guess at letters or clue numbers without verifying them visually
- **Be methodical**: Compare grids systematically, don't jump to conclusions
- **Verify letter counts**: If an answer doesn't match the expected length, you've made a reading error - go back and re-read
- **Double-check**: Look at the user's grid multiple times before declaring what's wrong
- **Be specific**: Say exactly which intersection is wrong (e.g., "39-Across SASS crosses 35-Down at the third letter")
- **Admit if unclear**: If the images are hard to read, ask the user for clarification
- **One error at a time**: Focus on identifying the actual errors shown by blue squares, not hypothetical ones

## Common Pitfalls to Avoid

1. **Rushing to conclusions** - Read every letter carefully before analyzing
2. **Guessing at clue numbers** - Verify clue numbers visually or ask the user
3. **Not reading complete words** - Always trace the full ACROSS and DOWN answers through error squares
4. **Suggesting impossible answers** - If letter counts don't match, you misread something
5. **Comparing partial grids** - Make sure you can see both complete grids before starting
6. **Making up crossing clues** - Only reference clues you can actually see or that the user confirms

## Output Format

When you find the error(s), report:
- The incorrect word(s) the user has
- The correct word(s) it should be
- The specific square(s) that are wrong (with the intersection if applicable)
- What letter the user has vs. what it should be
