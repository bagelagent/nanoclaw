#!/usr/bin/env node

/**
 * Migrate heartbeat tasks to post to Discord #bagel-heartbeat channel
 *
 * This updates all scheduled tasks to send their output to the Discord
 * #bagel-heartbeat channel instead of the main chat.
 */

import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.join(__dirname, '../../store/messages.db');
const HEARTBEAT_CHANNEL_JID = 'discord:1470289686554738761';

console.log('🔄 Migrating heartbeat tasks to Discord channel...');
console.log(`Database: ${DB_PATH}`);
console.log(`Target JID: ${HEARTBEAT_CHANNEL_JID}\n`);

const db = new Database(DB_PATH);

// Get current tasks
const tasks = db.prepare('SELECT * FROM scheduled_tasks WHERE status = "active"').all();

console.log(`Found ${tasks.length} active tasks:\n`);

tasks.forEach(task => {
  console.log(`📋 ${task.id}`);
  console.log(`   Prompt: ${task.prompt.substring(0, 60)}...`);
  console.log(`   Current JID: ${task.chat_jid}`);
  console.log(`   Group: ${task.group_folder}`);
  console.log(`   Schedule: ${task.schedule_type} ${task.schedule_value}`);
  console.log();
});

// Update all tasks to use the heartbeat channel JID
const updateStmt = db.prepare(
  'UPDATE scheduled_tasks SET chat_jid = ? WHERE status = "active"'
);

const result = updateStmt.run(HEARTBEAT_CHANNEL_JID);

console.log(`✅ Updated ${result.changes} tasks to use heartbeat channel JID`);
console.log(`\nAll heartbeat outputs will now post to #bagel-heartbeat on Discord! 🎉`);

db.close();
