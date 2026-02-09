# #bagel-heartbeat (abcd's server)

This is a Discord channel workspace with persistent memory.

## Heartbeat System

- Runs every 30 minutes automatically
- Extensible architecture for adding new activities
- See `README.md` and `heartbeat.js` for details
- To add new activities, create a class extending `HeartbeatActivity` and register it
