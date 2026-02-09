---
name: movie-times
description: Find movie showtimes at Cinemark Daly City for family-friendly films. Checks what's playing this weekend that would be appropriate for a 7-year-old.
---

# Movie Times Skill

Find family-friendly movies playing at Cinemark Daly City this weekend.

## User Context
- Location: Cinemark Daly City (Serramonte 16)
- Audience: User + Edith (7 years old daughter)
- Focus: Weekend showtimes for age-appropriate films

## Process

1. **Open Cinemark Website**
   ```bash
   agent-browser open "https://www.cinemark.com/theatres/ca-daly-city/cinemark-serramonte-16"
   ```

2. **Take Snapshot to See Page Structure**
   ```bash
   agent-browser snapshot -i
   ```

3. **Navigate to Showtimes**
   - Look for "Showtimes" or date selector
   - Select upcoming weekend dates
   - Use `agent-browser click [ref]` to interact

4. **Extract Movie Information**
   For each movie showing, collect:
   - Title
   - Rating (G, PG, PG-13 - filter out R-rated)
   - Genre
   - Showtimes
   - Direct booking link

5. **Filter for Family-Friendly**
   Prioritize:
   - G and PG rated films
   - PG-13 only if appropriate (animated, adventure, etc.)
   - Filter out: horror, mature themes, R-rated

6. **Present Results**
   Format as:
   ```
   🎬 Movie Times at Cinemark Daly City - This Weekend

   [MOVIE TITLE] (Rating)
   Genre: [genre]
   Showtimes: [times]
   🎟️ [Direct link to book tickets]

   Why it's good for Edith: [brief note if relevant]
   ```

## Tips
- Check both Saturday and Sunday showtimes
- Matinee times (before 5pm) are usually better for kids
- If you see limited listings, try clicking date selectors
- Some movies may require expanding to see all showtimes

## Fallback
If the Cinemark site doesn't load or is hard to navigate:
- Try Fandango: `https://www.fandango.com/cinemark-serramonte-16-aacoj/theater-page`
- Or Google: Search "Cinemark Daly City showtimes" and use agent-browser

## Example Output

```
🎬 This Weekend at Cinemark Daly City

**Moana 2** (PG)
Genre: Animation, Adventure, Musical
Saturday: 10:30am, 1:15pm, 4:00pm, 6:45pm
Sunday: 11:00am, 1:45pm, 4:30pm
🎟️ https://www.cinemark.com/[booking-link]

Perfect for Edith - animated musical adventure!

**Paddington in Peru** (PG)
Genre: Family, Adventure, Comedy
Saturday: 11:00am, 2:00pm, 5:00pm
Sunday: 12:00pm, 3:00pm, 6:00pm
🎟️ https://www.cinemark.com/[booking-link]

Great family film with the lovable bear!
```

## Notes
- Remember Edith is 7, so focus on G/PG content
- Weekend = This coming Saturday/Sunday
- Link directly to ticket purchase pages when possible
