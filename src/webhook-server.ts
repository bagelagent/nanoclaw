/**
 * GitHub Webhook Server
 * Listens for GitHub webhook events (issue assignments) and creates work assignments
 */

import express, { Request, Response } from 'express';
import crypto from 'crypto';
import { logger } from './logger.js';
import {
  createGitHubAssignment,
  getGitHubAssignmentByIssue,
  GitHubAssignment,
} from './db.js';
import { fetchIssue } from './github-api.js';
import { startWorkingOnIssue } from './github-issue-worker.js';

const app = express();
app.use(express.json());

interface GitHubWebhookPayload {
  action: string;
  issue: {
    number: number;
    title: string;
    body: string;
    html_url: string;
    labels: Array<{ name: string }>;
    assignee?: {
      login: string;
    };
    assignees: Array<{ login: string }>;
  };
  repository: {
    owner: {
      login: string;
    };
    name: string;
  };
  sender: {
    login: string;
  };
}

interface GitHubCommentWebhookPayload {
  action: string;
  issue: {
    number: number;
    title: string;
    html_url: string;
    assignees: Array<{ login: string }>;
  };
  comment: {
    id: number;
    body: string;
    user: {
      login: string;
    };
    created_at: string;
  };
  repository: {
    owner: {
      login: string;
    };
    name: string;
  };
  sender: {
    login: string;
  };
}

/**
 * Verify GitHub webhook signature
 */
function verifyGitHubSignature(
  payload: string,
  signature: string | undefined,
  secret: string,
): boolean {
  if (!signature) {
    return false;
  }

  const hmac = crypto.createHmac('sha256', secret);
  const digest = 'sha256=' + hmac.update(payload).digest('hex');
  return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(digest));
}

/**
 * Handle GitHub issue webhook events
 */
async function handleIssueWebhook(payload: GitHubWebhookPayload) {
  const { action, issue, repository, sender } = payload;

  logger.info(
    {
      action,
      issue: issue.number,
      repo: `${repository.owner.login}/${repository.name}`,
      assignees: issue.assignees.map((a) => a.login),
    },
    'GitHub webhook: issue event',
  );

  // Only process 'assigned' events
  if (action !== 'assigned') {
    logger.debug({ action }, 'Ignoring non-assigned issue event');
    return;
  }

  // Check if the bot is assigned to this issue
  const botUsername = process.env.GITHUB_BOT_USERNAME || 'bagel-bot';
  const isAssignedToBot = issue.assignees.some((a) => a.login === botUsername);

  if (!isAssignedToBot) {
    logger.debug(
      { assignees: issue.assignees.map((a) => a.login), botUsername },
      'Issue not assigned to bot',
    );
    return;
  }

  // Check if assignment already exists
  const existing = getGitHubAssignmentByIssue(
    repository.owner.login,
    repository.name,
    issue.number,
  );

  if (existing) {
    logger.info(
      { assignmentId: existing.id, status: existing.status },
      'Assignment already exists',
    );
    return;
  }

  // Create new assignment
  const assignmentId = createGitHubAssignment({
    issue_url: issue.html_url,
    repo_owner: repository.owner.login,
    repo_name: repository.name,
    issue_number: issue.number,
    title: issue.title,
    description: issue.body || null,
    labels: issue.labels.length > 0 ? JSON.stringify(issue.labels.map((l) => l.name)) : null,
    assigned_by: sender.login,
    assigned_at: new Date().toISOString(),
  });

  logger.info(
    {
      assignmentId,
      issue: issue.number,
      repo: `${repository.owner.login}/${repository.name}`,
      title: issue.title,
    },
    'Created GitHub assignment from webhook',
  );

  // Get the full assignment record and start working on it immediately
  const fullAssignment = getGitHubAssignmentByIssue(
    repository.owner.login,
    repository.name,
    issue.number,
  );

  if (fullAssignment) {
    // Start working on the issue in the background (don't await)
    // This allows the webhook to respond quickly
    startWorkingOnIssue(fullAssignment).catch((err) => {
      logger.error({ err, assignmentId }, 'Background issue worker failed');
    });
  }
}

/**
 * Handle GitHub issue comment webhook events
 */
async function handleIssueCommentWebhook(payload: GitHubCommentWebhookPayload) {
  const { action, issue, comment, repository, sender } = payload;

  logger.info(
    {
      action,
      issue: issue.number,
      repo: `${repository.owner.login}/${repository.name}`,
      commenter: comment.user.login,
    },
    'GitHub webhook: issue_comment event',
  );

  // Only process 'created' comments
  if (action !== 'created') {
    logger.debug({ action }, 'Ignoring non-created comment event');
    return;
  }

  // Check if this issue has an assignment
  const assignment = getGitHubAssignmentByIssue(
    repository.owner.login,
    repository.name,
    issue.number,
  );

  if (!assignment) {
    logger.debug({ issue: issue.number }, 'No assignment found for this issue');
    return;
  }

  // Check if the comment is from the bot itself (ignore)
  const botUsername = process.env.GITHUB_BOT_USERNAME || 'bagel-bot';
  if (comment.user.login === botUsername) {
    logger.debug('Ignoring comment from bot itself');
    return;
  }

  // Check for approval keywords in the comment
  const approvalKeywords = ['approved', 'lgtm', 'go ahead', 'looks good', 'ship it'];
  const commentLower = comment.body.toLowerCase();
  const isApproval = approvalKeywords.some((keyword) => commentLower.includes(keyword));

  if (isApproval) {
    logger.info(
      {
        assignmentId: assignment.id,
        issue: issue.number,
        approver: comment.user.login,
      },
      'Approval detected in comment - agent should proceed with implementation',
    );

    // The agent is polling for comments and will detect this approval
    // We don't need to do anything here - just log it
  }
}

/**
 * Start the webhook server
 */
export function startWebhookServer(port: number = 3000) {
  const webhookSecret = process.env.GITHUB_WEBHOOK_SECRET;

  if (!webhookSecret) {
    logger.warn('GITHUB_WEBHOOK_SECRET not set - webhook signature validation disabled');
  }

  // Health check endpoint
  app.get('/health', (req: Request, res: Response) => {
    res.json({ status: 'ok', service: 'nanoclaw-webhooks' });
  });

  // GitHub webhook endpoint
  app.post('/webhooks/github', async (req: Request, res: Response) => {
    const signature = req.headers['x-hub-signature-256'] as string | undefined;
    const event = req.headers['x-github-event'] as string | undefined;
    const payload = JSON.stringify(req.body);

    // Verify signature if secret is configured
    if (webhookSecret && !verifyGitHubSignature(payload, signature, webhookSecret)) {
      logger.warn({ event, signature }, 'Invalid webhook signature');
      res.status(401).json({ error: 'Invalid signature' });
      return;
    }

    // Handle different event types
    if (event === 'issues') {
      try {
        await handleIssueWebhook(req.body as GitHubWebhookPayload);
        res.json({ status: 'ok' });
      } catch (err) {
        logger.error({ err, event }, 'Error handling webhook');
        res.status(500).json({ error: 'Internal server error' });
      }
    } else if (event === 'issue_comment') {
      try {
        await handleIssueCommentWebhook(req.body as GitHubCommentWebhookPayload);
        res.json({ status: 'ok' });
      } catch (err) {
        logger.error({ err, event }, 'Error handling webhook');
        res.status(500).json({ error: 'Internal server error' });
      }
    } else if (event === 'ping') {
      logger.info('Received GitHub webhook ping');
      res.json({ status: 'ok', message: 'pong' });
    } else {
      logger.debug({ event }, 'Ignoring webhook event');
      res.json({ status: 'ignored' });
    }
  });

  // Start server
  app.listen(port, () => {
    logger.info({ port }, 'GitHub webhook server started');
    console.log(`\n🪝 Webhook server listening on http://localhost:${port}`);
    console.log(`   Endpoint: POST http://localhost:${port}/webhooks/github\n`);
  });

  return app;
}
