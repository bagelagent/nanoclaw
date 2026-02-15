# NanoClaw Cleanup System

Automatic cleanup to prevent disk space issues.

## What Gets Cleaned

1. **Container logs** older than 7 days (`/workspace/logs/container-*.log`)
2. **Unused containers** stopped for more than 7 days
3. **Dangling images** (untagged and unreferenced)

## Installation

Run on the host machine:

```bash
cd /workspace/project/scripts
./install-cleanup-cron.sh
```

This adds a cron job that runs daily at 3 AM.

## Manual Cleanup

To run cleanup manually:

```bash
/workspace/project/scripts/cleanup-old-data.sh
```

## Logs

Cleanup logs are written to: `/workspace/logs/cleanup.log`

## Configuration

To adjust retention period, edit `cleanup-old-data.sh`:

```bash
LOG_RETENTION_DAYS=7  # Change this value
```

## Checking Cron Status

```bash
crontab -l | grep nanoclaw
```

## Removing Cron Job

```bash
crontab -l | grep -v "cleanup-old-data.sh" | crontab -
```
