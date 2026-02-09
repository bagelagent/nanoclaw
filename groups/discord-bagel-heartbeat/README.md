# #bagel-heartbeat System

This workspace contains the heartbeat system for monitoring and maintaining the Discord channel.

## Heartbeat System

The heartbeat runs automatically every 30 minutes and executes registered activities.

### Architecture

The system is built with extensibility in mind:

- **HeartbeatActivity**: Base class for all activities
- **HeartbeatManager**: Orchestrates activity execution
- **Activity Classes**: Individual tasks that run during each heartbeat

### Adding New Activities

To add a new heartbeat activity:

1. Create a class that extends `HeartbeatActivity`
2. Implement the `execute()` method
3. Optionally override `shouldRun()` for conditional execution
4. Register it in the `main()` function

Example:

```javascript
class MyActivity extends HeartbeatActivity {
  constructor() {
    super('my-activity', 'Does something useful');
  }

  async execute() {
    // Your logic here
    return {
      success: true,
      message: 'Activity completed',
      data: { /* optional data */ }
    };
  }
}

// In main():
manager.registerActivity(new MyActivity());
```

### Controlling Activity Frequency

Activities can run at different intervals:

```javascript
class LessFrequentActivity extends HeartbeatActivity {
  constructor() {
    super('infrequent', 'Runs less often');
    this.runEveryNBeats = 6; // Every 3 hours (30min * 6)
    this.beatCount = 0;
  }

  shouldRun() {
    if (!this.enabled) return false;
    this.beatCount++;
    return this.beatCount % this.runEveryNBeats === 0;
  }
}
```

### Current Activities

1. **health-check**: Performs system health checks (runs every 2 hours)

### Files

- `heartbeat.js`: Main heartbeat system implementation
- `logs/`: Heartbeat execution logs (auto-generated)

### Manual Execution

To test the heartbeat manually:

```bash
cd /workspace/group
node heartbeat.js
```

### Scheduled Execution

The heartbeat runs automatically every 30 minutes via scheduled task.
