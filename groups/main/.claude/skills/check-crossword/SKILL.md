---
name: check-crossword
description: Find errors in crossword puzzles by comparing the user's grid to the NYT solution via API. Use when user asks to check their crossword, find errors, or needs help with the NYT crossword.
---

# Check Crossword Skill

Compare the user's NYT crossword grid against the official solution using the NYT API. No screenshots needed.

## Setup

- NYT-S cookie stored at: `/workspace/group/.nyt-cookie`
- Supports: Daily crossword, Midi, Mini
- Supports any date — today or past dates (e.g., "check my Saturday crossword", "check March 15 midi")

## Process

1. **Determine which puzzle and date** — ask the user if unclear.
   - Default puzzle type: daily crossword
   - Default date: today
   - If the user says "yesterday's" or "Saturday's" or gives a specific date, use that date
   - Format date as YYYY-MM-DD for API calls

2. **Get the puzzle ID and solution:**
   ```bash
   NYT_S=$(cat /workspace/group/.nyt-cookie)
   # For daily:
   curl -s -H "Cookie: NYT-S=$NYT_S" "https://www.nytimes.com/svc/crosswords/v2/puzzle/daily-YYYY-MM-DD.json"
   # For midi:
   curl -s -H "Cookie: NYT-S=$NYT_S" "https://www.nytimes.com/svc/crosswords/v2/puzzle/midi-YYYY-MM-DD.json"
   # For mini:
   curl -s -H "Cookie: NYT-S=$NYT_S" "https://www.nytimes.com/svc/crosswords/v2/puzzle/mini-YYYY-MM-DD.json"
   ```
   The response contains `results[0].puzzle_id`, `results[0].puzzle_meta` (width, height), and `results[0].puzzle_data` (answers array, layout array).

3. **Get the user's game state:**
   ```bash
   NYT_S=$(cat /workspace/group/.nyt-cookie)
   PUZZLE_ID=<from step 2>
   # For daily:
   curl -s -H "Cookie: NYT-S=$NYT_S" "https://www.nytimes.com/svc/crosswords/v6/game/$PUZZLE_ID.json"
   # Response has: board.cells[i].guess

   # For midi:
   curl -s -H "Cookie: NYT-S=$NYT_S" "https://www.nytimes.com/svc/games/state/crossword_midi/latests?puzzle_ids=$PUZZLE_ID"
   # Response has: states[0].game_data.cells (object with string keys like "0", "1", etc.)

   # For mini:
   curl -s -H "Cookie: NYT-S=$NYT_S" "https://www.nytimes.com/svc/games/state/crossword_mini/latests?puzzle_ids=$PUZZLE_ID"
   ```

4. **Compare programmatically** using Python:
   ```python
   # Solution: answers is a flat list, layout indicates black squares (0) vs letter cells
   # Game state: cells contain the user's guesses
   # Compare answer[i] vs user_guess[i] for each non-black cell
   ```

5. **Report errors** via send_message with:
   - Which row/col has the error
   - What letter the user has vs what it should be
   - The across and down clues that intersect at that cell (from puzzle_data.clues)

## Important Notes

- **Blue highlighted squares in screenshots are the CURSOR, NOT errors.** Ignore them.
- **All API calls are read-only GETs.** They do NOT affect the user's streak, star, or solve status.
- **Cookie location:** `/workspace/group/.nyt-cookie` — if it expires, ask the user for a new NYT-S cookie.
- **If the API returns empty game state** (calcs: {}), the endpoint format may be wrong. Daily uses v6/game, Midi/Mini use /svc/games/state/.
- **Date format:** YYYY-MM-DD for puzzle URLs.
- **The user has an almost 3-year streak.** Never do anything that could affect it — only read data, never POST/PUT.
- If the user sends a screenshot instead, you can still try the API approach first (it's more reliable). Only fall back to visual comparison if the API fails.

## Puzzle Data Structure

**Solution (v2/puzzle):**
- `results[0].puzzle_data.answers` — flat list of characters (length = width × height)
- `results[0].puzzle_data.layout` — flat list where 0 = black square, non-zero = letter cell
- `results[0].puzzle_data.clues.A` — across clues
- `results[0].puzzle_data.clues.D` — down clues

**Daily game state (v6/game):**
- `board.cells[i].guess` — user's letter for cell i
- `board.cells[i].blank` — true for black squares
- `calcs.solved` — whether puzzle is complete
- `calcs.secondsSpentSolving` — solve time

**Midi/Mini game state (games/state):**
- `states[0].game_data.cells` — object with string index keys ("0", "1", etc.) mapping to letters
