/**
 * GitHub Webhook Server
 * Listens for GitHub webhook events (issue assignments) and creates work assignments
 */

import express, { Request, Response } from 'express';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { logger } from './logger.js';
import {
  createGitHubAssignment,
  getGitHubAssignmentByIssue,
  GitHubAssignment,
  getAllRegisteredGroups,
  getRegisteredGroup,
  setRegisteredGroup,
} from './db.js';
import { RegisteredGroup } from './types.js';
import { fetchIssue } from './github-api.js';
import { runContainerAgent } from './container-runner.js';
import { GROUPS_DIR } from './config.js';

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

interface GitHubPRWebhookPayload {
  action: string;
  pull_request: {
    number: number;
    title: string;
    body: string | null;
    html_url: string;
    state: string;
    assignees: Array<{ login: string }>;
    user: {
      login: string;
    };
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

interface GitHubPRReviewWebhookPayload {
  action: string;
  pull_request: {
    number: number;
    title: string;
    html_url: string;
  };
  review: {
    id: number;
    body: string | null;
    state: string; // 'approved', 'changes_requested', 'commented'
    user: {
      login: string;
    };
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
 * Get or create a GitHub group for a repository
 */
function getOrCreateGitHubGroup(repoOwner: string, repoName: string): RegisteredGroup {
  const groupKey = `github-${repoOwner}-${repoName}`;
  const folderName = `github-${repoOwner}-${repoName}`;

  // Check if group already exists
  let group = getRegisteredGroup(groupKey);

  if (!group) {
    // Create new GitHub group
    const groupFolder = path.join(GROUPS_DIR, folderName);

    // Create group directory if it doesn't exist
    if (!fs.existsSync(groupFolder)) {
      fs.mkdirSync(groupFolder, { recursive: true });
      logger.info({ folder: folderName }, 'Created new GitHub group folder');

      // Create initial CLAUDE.md
      const claudeMd = `# GitHub: ${repoOwner}/${repoName}

This is an auto-created group for handling GitHub webhooks for the repository **${repoOwner}/${repoName}**.

## Workspace

The repository is cloned to \`/workspace/group/${repoName}/\` and persists across container runs.

## Memory

Conversation history and context for this repository's issues and PRs are stored here.
`;
      fs.writeFileSync(path.join(groupFolder, 'CLAUDE.md'), claudeMd);
    }

    // Register the group
    const newGroup = {
      name: `GitHub: ${repoOwner}/${repoName}`,
      folder: folderName,
      trigger: '@Bagel', // Not used for GitHub groups
      added_at: new Date().toISOString(),
      jid: groupKey, // Use groupKey as JID for GitHub groups
    };

    setRegisteredGroup(groupKey, newGroup);
    logger.info({ groupKey, folder: folderName }, 'Registered new GitHub group');

    group = newGroup;
  }

  return group!
}

/**
 * Spawn a container to handle GitHub webhook events
 */
function spawnGitHubContainer(eventType: string, data: any) {
  // Get or create GitHub group for this repository
  const group = getOrCreateGitHubGroup(data.repo_owner, data.repo_name);

  // Build prompt based on event type
  let prompt = '';
  if (eventType === 'issue_assigned') {
    prompt = buildIssueAssignedPrompt(data);
  } else if (eventType === 'plan_approved') {
    prompt = buildPlanApprovedPrompt(data);
  } else {
    logger.warn({ eventType }, 'Unknown GitHub event type');
    return;
  }

  const chatJid = `github-${eventType}-${data.issue_number}-${Date.now()}`;
  const sessionId = `github-${data.assignmentId || data.issue_number}-${Date.now()}`;

  // Spawn container in background (don't await)
  runContainerAgent(
    group,
    {
      prompt,
      sessionId,
      groupFolder: group.folder,
      chatJid,
      isMain: false, // GitHub groups are not main
    },
    () => {}, // No-op callback
  ).catch((err) => {
    logger.error({ err, eventType, issue: data.issue_number || data.pr_number }, 'GitHub container failed');
  });

  logger.info({ eventType, issue: data.issue_number || data.pr_number, chatJid, groupFolder: group.folder }, 'Spawned GitHub container');
}

/**
 * Build prompt for issue_assigned event
 */
function buildIssueAssignedPrompt(data: any): string {
  return `You've been assigned to work on a GitHub issue. Here are the details:

**Repository**: ${data.repo_owner}/${data.repo_name}
**Issue #${data.issue_number}**: ${data.title}
**URL**: ${data.issue_url}

**Description**:
${data.description || '(No description provided)'}

---

**WORKSPACE**: The repository is in \`/workspace/group/${data.repo_name}/\`

**YOUR WORKFLOW**:

1. **Post initial comment** on the issue saying you've started work

2. **Get the repository**:
   - First check if \`/workspace/group/${data.repo_name}/\` exists (use Bash: \`ls -la /workspace/group/${data.repo_name} 2>&1\`)
   - If it exists: \`cd\` into it and \`git pull\` to update
   - If it doesn't exist: Use github_clone_repo to clone it (it will go to /workspace/project/github-work/, then you should move it to /workspace/group/)

3. **Explore the codebase** thoroughly to understand the issue

4. **Enter plan mode** (use EnterPlanMode) to design your solution

5. **Post your plan** as a comment on issue #${data.issue_number} using github_comment

6. **Stop and wait** - DO NOT implement yet. Your work is done for now.

A new container will be spawned when the user approves your plan.

**Important**:
- The repository persists in /workspace/group/ across runs - don't clone unnecessarily
- Focus on creating a thorough, well-thought-out plan
- Be specific about what files you'll change and why
- Don't implement anything in this phase - just plan

Begin by posting an initial comment and checking if the repository exists.`;
}

/**
 * Build prompt for plan_approved event
 */
function buildPlanApprovedPrompt(data: any): string {
  return `Your plan for GitHub issue #${data.issue_number} has been approved by ${data.approver}!

**Repository**: ${data.repo_owner}/${data.repo_name}
**Issue**: ${data.title}
**URL**: ${data.issue_url}

---

**WORKSPACE**: The repository should be in \`/workspace/group/${data.repo_name}/\`

**YOUR WORKFLOW**:

1. **Get the repository**:
   - \`cd /workspace/group/${data.repo_name}\` (it should already exist from the planning phase)
   - Run \`git pull\` to get latest changes
   - If it somehow doesn't exist, clone it first

2. **Read your plan** - Use github_get_comments to see what you previously proposed

3. **Implement the solution**:
   - Create a new branch: github_create_branch in /workspace/group/${data.repo_name} (format: "bagel/issue-${data.issue_number}-brief-description")
   - Make your code changes using Edit/Write tools
   - Follow the plan you created
   - Test your changes if possible

4. **Commit and push** using github_commit_push with a clear message

5. **Create a pull request** using github_create_pr:
   - Title should reference the issue
   - Body should describe the changes
   - Include "Fixes #${data.issue_number}"

6. **Comment on the issue** with the PR link using github_comment

**Important**:
- The repository persists in /workspace/group/${data.repo_name}/ - use it directly
- Follow the plan you created earlier
- Write clean, tested code
- Create a professional PR ready for review

Begin by navigating to the repository and reviewing your approved plan.`;
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

  // Spawn a container to handle this issue assignment
  spawnGitHubContainer('issue_assigned', {
    assignmentId,
    repo_owner: repository.owner.login,
    repo_name: repository.name,
    issue_number: issue.number,
    title: issue.title,
    description: issue.body || '',
    issue_url: issue.html_url,
  });
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
      'Approval detected in comment - spawning container to implement',
    );

    // Spawn a container to handle the approved implementation
    spawnGitHubContainer('plan_approved', {
      assignmentId: assignment.id,
      repo_owner: repository.owner.login,
      repo_name: repository.name,
      issue_number: issue.number,
      title: issue.title,
      issue_url: issue.html_url,
      approver: comment.user.login,
    });
  }
}

/**
 * Handle GitHub PR webhook events
 */
async function handlePRWebhook(payload: GitHubPRWebhookPayload) {
  const { action, pull_request, repository, sender } = payload;

  logger.info(
    {
      action,
      pr: pull_request.number,
      repo: `${repository.owner.login}/${repository.name}`,
      author: pull_request.user.login,
    },
    'GitHub webhook: pull_request event',
  );

  // Handle relevant PR actions (opened, assigned, etc.)
  // For now, just log them
  // Can add custom logic later (e.g., spawn container for certain actions)
}

/**
 * Handle GitHub PR review webhook events
 */
async function handlePRReviewWebhook(payload: GitHubPRReviewWebhookPayload) {
  const { action, pull_request, review, repository, sender } = payload;

  logger.info(
    {
      action,
      pr: pull_request.number,
      repo: `${repository.owner.login}/${repository.name}`,
      reviewer: review.user.login,
      review_state: review.state,
    },
    'GitHub webhook: pull_request_review event',
  );

  // Handle PR reviews (approved, changes_requested, etc.)
  // For now, just log them
  // Can add custom logic later (e.g., spawn container to address review comments)
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
    } else if (event === 'pull_request') {
      try {
        await handlePRWebhook(req.body as GitHubPRWebhookPayload);
        res.json({ status: 'ok' });
      } catch (err) {
        logger.error({ err, event }, 'Error handling webhook');
        res.status(500).json({ error: 'Internal server error' });
      }
    } else if (event === 'pull_request_review') {
      try {
        await handlePRReviewWebhook(req.body as GitHubPRReviewWebhookPayload);
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
