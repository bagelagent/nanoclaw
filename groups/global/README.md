# Global Research Queue System

A persistent research system for Bagel to continuously learn and expand knowledge.

## Overview

The research queue is a global knowledge-building system where:
- Topics can be added from any group
- Research happens automatically via scheduled heartbeat
- Results are documented and searchable
- New topics emerge from completed research

## Files

- **research-queue.json** - The global queue (pending + completed topics)
- **research/** - Completed research documents (one per topic)
- **scripts/add-research-topic.sh** - Helper to add topics
- **scripts/research-heartbeat.sh** - Legacy shell script (not used - agent does research directly)

## Queue Structure

```json
{
  "queue": [
    {
      "id": "rq-{timestamp}-{random}",
      "topic": "Research question",
      "source": "Where this topic came from",
      "addedBy": "group-folder",
      "addedAt": "ISO timestamp",
      "priority": 1-10,
      "status": "pending|in-progress|completed",
      "tags": ["tag1", "tag2"]
    }
  ],
  "completed": [
    {
      "id": "...",
      "topic": "...",
      "completedAt": "...",
      "summary": "Brief summary",
      "sources": ["url1", "url2"]
    }
  ]
}
```

## Priority Levels

- **10**: Urgent/time-sensitive
- **7-9**: High priority - interesting/important
- **4-6**: Medium priority - general interest
- **1-3**: Low priority - nice to know

## How to Add Topics

### From Command Line
```bash
/workspace/project/groups/global/scripts/add-research-topic.sh \
  "Your research question" \
  "source/context" \
  "tag1,tag2,tag3" \
  priority
```

### From Agent (any group)
```bash
# Read current queue
cat /workspace/project/groups/global/research-queue.json

# Add new topic
/workspace/project/groups/global/scripts/add-research-topic.sh \
  "Topic" "source" "tags" priority

# Or edit JSON directly if you need more control
```

## Heartbeat Schedule

Research runs every **6 hours** (21600000ms) in isolated mode.

Each session:
1. Picks highest priority pending topic
2. Researches thoroughly (web search, article reading, synthesis)
3. Documents findings in `/workspace/project/groups/global/research/{id}.md`
4. Updates queue (move to completed, add summary)
5. May add new topics discovered during research

## Research Output

Each research file contains:
- **Summary**: 2-3 paragraph overview
- **Key Findings**: Bullet points of main insights
- **Deep Dive**: Detailed exploration
- **Connections**: Links to other knowledge
- **Follow-up Questions**: New research topics
- **Sources**: Cited references with URLs

## Viewing Research

```bash
# List all research files
ls -lt /workspace/project/groups/global/research/

# Read a specific research
cat /workspace/project/groups/global/research/rq-*.md

# Check queue status
cat /workspace/project/groups/global/research-queue.json | jq '.queue[] | select(.status == "pending") | .topic'
```

## Next Run

Check scheduled tasks:
```bash
# From main group
/workspace/project/data/ipc/main/current_tasks.json
```

Research heartbeat task ID: `1770753436698-or3db4`
