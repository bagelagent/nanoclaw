/**
 * Heartbeat System for #bagel-heartbeat
 *
 * A modular, extensible heartbeat system that runs periodic tasks.
 * Add new heartbeat activities by implementing the HeartbeatActivity interface.
 */

// ============================================================================
// Heartbeat Activity Interface
// ============================================================================

/**
 * Base class for heartbeat activities.
 * Each activity should extend this and implement the execute() method.
 */
class HeartbeatActivity {
  constructor(name, description) {
    this.name = name;
    this.description = description;
    this.enabled = true;
    this.lastRun = null;
    this.lastRunStatus = null;
  }

  /**
   * Execute the heartbeat activity.
   * Should return an object with { success: boolean, message: string, data?: any }
   */
  async execute() {
    throw new Error('execute() must be implemented by subclass');
  }

  /**
   * Optional: Check if this activity should run this cycle
   * Useful for activities that should run less frequently than the heartbeat interval
   */
  shouldRun() {
    return this.enabled;
  }

  /**
   * Record the result of a run
   */
  recordRun(result) {
    this.lastRun = new Date().toISOString();
    this.lastRunStatus = result;
  }
}

// ============================================================================
// Example Heartbeat Activities
// ============================================================================

/**
 * Example: Periodic health check
 */
class HealthCheckActivity extends HeartbeatActivity {
  constructor() {
    super(
      'health-check',
      'Performs system health checks'
    );
    this.runEveryNBeats = 4; // Run every 2 hours (30min * 4)
    this.beatCount = 0;
  }

  shouldRun() {
    if (!this.enabled) return false;
    this.beatCount++;
    return this.beatCount % this.runEveryNBeats === 0;
  }

  async execute() {
    try {
      // TODO: Implement health check logic

      return {
        success: true,
        message: 'Health check passed',
        data: {
          timestamp: new Date().toISOString(),
          status: 'healthy'
        }
      };
    } catch (error) {
      return {
        success: false,
        message: `Health check failed: ${error.message}`
      };
    }
  }
}

// ============================================================================
// Heartbeat Manager
// ============================================================================

class HeartbeatManager {
  constructor() {
    this.activities = new Map();
    this.running = false;
    this.beatCount = 0;
    this.startTime = null;
  }

  /**
   * Register a heartbeat activity
   */
  registerActivity(activity) {
    if (!(activity instanceof HeartbeatActivity)) {
      throw new Error('Activity must be an instance of HeartbeatActivity');
    }
    this.activities.set(activity.name, activity);
    console.log(`[Heartbeat] Registered activity: ${activity.name}`);
  }

  /**
   * Unregister a heartbeat activity
   */
  unregisterActivity(name) {
    this.activities.delete(name);
    console.log(`[Heartbeat] Unregistered activity: ${name}`);
  }

  /**
   * Enable/disable a specific activity
   */
  setActivityEnabled(name, enabled) {
    const activity = this.activities.get(name);
    if (activity) {
      activity.enabled = enabled;
      console.log(`[Heartbeat] Activity ${name} ${enabled ? 'enabled' : 'disabled'}`);
    }
  }

  /**
   * Execute a single heartbeat cycle
   */
  async beat() {
    this.beatCount++;
    const beatNumber = this.beatCount;
    const timestamp = new Date().toISOString();

    console.log(`\n[Heartbeat] Beat #${beatNumber} at ${timestamp}`);
    console.log(`[Heartbeat] Running ${this.activities.size} registered activities`);

    const results = [];

    for (const [name, activity] of this.activities) {
      if (!activity.shouldRun()) {
        console.log(`[Heartbeat] Skipping ${name} (shouldRun returned false)`);
        continue;
      }

      console.log(`[Heartbeat] Executing ${name}...`);
      try {
        const result = await activity.execute();
        activity.recordRun(result);
        results.push({ name, ...result });

        const status = result.success ? '✓' : '✗';
        console.log(`[Heartbeat] ${status} ${name}: ${result.message}`);
      } catch (error) {
        const result = {
          success: false,
          message: `Unhandled error: ${error.message}`
        };
        activity.recordRun(result);
        results.push({ name, ...result });
        console.error(`[Heartbeat] ✗ ${name}: ${error.message}`);
      }
    }

    return {
      beatNumber,
      timestamp,
      results,
      summary: {
        total: results.length,
        successful: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length
      }
    };
  }

  /**
   * Get status of all activities
   */
  getStatus() {
    const activities = Array.from(this.activities.values()).map(activity => ({
      name: activity.name,
      description: activity.description,
      enabled: activity.enabled,
      lastRun: activity.lastRun,
      lastRunStatus: activity.lastRunStatus
    }));

    return {
      running: this.running,
      beatCount: this.beatCount,
      startTime: this.startTime,
      uptime: this.startTime ? Date.now() - new Date(this.startTime).getTime() : 0,
      activities
    };
  }
}

// ============================================================================
// Main Execution
// ============================================================================

async function main() {
  console.log('[Heartbeat] Starting heartbeat system...');

  const manager = new HeartbeatManager();
  manager.startTime = new Date().toISOString();
  manager.running = true;

  // Register activities
  manager.registerActivity(new HealthCheckActivity());

  // Execute the heartbeat
  const result = await manager.beat();

  // Log summary
  console.log('\n[Heartbeat] Beat complete');
  console.log(`[Heartbeat] Summary: ${result.summary.successful}/${result.summary.total} successful`);

  return result;
}

// Run if executed directly
if (require.main === module) {
  main()
    .then(result => {
      console.log('\n[Heartbeat] Heartbeat execution finished');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n[Heartbeat] Fatal error:', error);
      process.exit(1);
    });
}

// Export for use as a module
module.exports = {
  HeartbeatActivity,
  HeartbeatManager,
  HealthCheckActivity,
  main
};
