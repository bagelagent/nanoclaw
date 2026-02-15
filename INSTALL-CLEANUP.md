# Install Cleanup Cron Job

The automatic cleanup system is ready, but needs to be installed from the **host machine** (not from inside a container).

## Installation Steps

SSH into your host machine and run:

```bash
cd /workspace/project/scripts
./install-cleanup-cron.sh
```

This will add a cron job that runs daily at 3 AM to clean up:
- Container logs older than 7 days
- Unused/stopped containers (7+ days old)
- Dangling Docker/Podman images

## Verify Installation

Check that the cron job was added:

```bash
crontab -l | grep cleanup
```

You should see:
```
# NanoClaw cleanup - runs daily at 3 AM
0 3 * * * /workspace/project/scripts/cleanup-old-data.sh >> /workspace/logs/cleanup.log 2>&1
```

## Test the Cleanup

You can run it manually anytime:

```bash
/workspace/project/scripts/cleanup-old-data.sh
```

## Monitoring

Check cleanup logs:

```bash
tail -f /workspace/logs/cleanup.log
```

## Troubleshooting

If crontab isn't working, you can also use systemd timers or just run the cleanup script manually from time to time.

The script is safe to run repeatedly - it only removes old data, never active files.
