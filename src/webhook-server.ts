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
  getRegisteredGroup,
  setRegisteredGroup,
} from './db.js';
import { RegisteredGroup } from './types.js';
import { fetchIssue, reactToComment, deleteReaction } from './github-api.js';
import { runContainerAgent } from './container-runner.js';
import { GROUPS_DIR } from './config.js';
import { GroupQueue } from './group-queue.js';

const app = express();

// Module-level reference to the queue, set during startWebhookServer()
let queue: GroupQueue | null = null;
app.use(express.json());

// Dedup: prevent duplicate container spawns from near-simultaneous webhooks
// Key = "eventType-owner/repo-issueNumber", value = timestamp
const recentEvents = new Map<string, number>();
const DEDUP_WINDOW_MS = 60_000; // 60 seconds

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

    // Register the group with a longer timeout for codebase exploration + planning
    const newGroup = {
      name: `GitHub: ${repoOwner}/${repoName}`,
      folder: folderName,
      trigger: '@Bagel', // Not used for GitHub groups
      added_at: new Date().toISOString(),
      jid: groupKey,
      containerConfig: {
        timeout: 900000, // 15 minutes — GitHub tasks involve deep codebase exploration
      },
    };

    setRegisteredGroup(groupKey, newGroup);
    logger.info({ groupKey, folder: folderName }, 'Registered new GitHub group');

    group = newGroup;
  }

  return group!
}

interface EmojiSwap {
  owner: string;
  repo: string;
  commentId: number;
  eyesReactionId: number;
}

/**
 * Spawn a container to handle GitHub webhook events.
 * Routes through GroupQueue to serialize queries per repo group —
 * multiple issues for the same repo run one at a time instead of
 * clobbering each other's pendingResolve callbacks.
 */
function spawnGitHubContainer(eventType: string, data: any, emojiSwap?: EmojiSwap) {
  // Dedup check: skip if we recently spawned for the same event+issue
  const dedupKey = `${eventType}-${data.repo_owner}/${data.repo_name}-${data.issue_number}`;
  const now = Date.now();
  const lastSeen = recentEvents.get(dedupKey);
  if (lastSeen && now - lastSeen < DEDUP_WINDOW_MS) {
    logger.info({ dedupKey }, 'Duplicate GitHub event within dedup window, skipping');
    return;
  }
  recentEvents.set(dedupKey, now);

  // Clean up old entries to prevent unbounded growth
  for (const [key, ts] of recentEvents) {
    if (now - ts > DEDUP_WINDOW_MS) recentEvents.delete(key);
  }

  // Get or create GitHub group for this repository
  const group = getOrCreateGitHubGroup(data.repo_owner, data.repo_name);

  // Build prompt based on event type
  let prompt = '';
  if (eventType === 'issue_assigned') {
    prompt = buildIssueAssignedPrompt(data);
  } else if (eventType === 'plan_approved') {
    prompt = buildPlanApprovedPrompt(data);
  } else if (eventType === 'question_answered') {
    prompt = buildQuestionAnsweredPrompt(data);
  } else {
    logger.warn({ eventType }, 'Unknown GitHub event type');
    return;
  }

  const chatJid = `github-${eventType}-${data.issue_number}-${Date.now()}`;
  const sessionId = `github-${data.assignmentId || data.issue_number}-${Date.now()}`;
  const taskId = `github-${eventType}-${data.repo_owner}-${data.repo_name}-${data.issue_number}-${Date.now()}`;

  // Use the group's registration key as the queue key (matches getOrCreateGitHubGroup)
  const groupKey = `github-${data.repo_owner}-${data.repo_name}`;

  const taskFn = async () => {
    try {
      await runContainerAgent(
        group,
        {
          prompt,
          sessionId,
          groupFolder: group.folder,
          chatJid,
          isMain: false,
        },
        (proc, containerName) => {
          if (queue) queue.registerProcess(groupKey, proc, containerName);
        },
      );
    } catch (err) {
      logger.error({ err, eventType, issue: data.issue_number || data.pr_number }, 'GitHub container failed');
    } finally {
      if (emojiSwap) {
        try {
          await deleteReaction(emojiSwap.owner, emojiSwap.repo, emojiSwap.commentId, emojiSwap.eyesReactionId);
        } catch (err) {
          logger.warn({ err, commentId: emojiSwap.commentId }, 'Failed to remove eyes reaction');
        }
        try {
          await reactToComment(emojiSwap.owner, emojiSwap.repo, emojiSwap.commentId, 'hooray');
        } catch (err) {
          logger.warn({ err, commentId: emojiSwap.commentId }, 'Failed to add hooray reaction');
        }
      }
    }
  };

  if (queue) {
    queue.enqueueTask(groupKey, taskId, taskFn);
  } else {
    // Fallback: no queue available (shouldn't happen in normal operation)
    logger.warn({ eventType }, 'No GroupQueue available, running GitHub container directly');
    taskFn();
  }

  logger.info({ eventType, issue: data.issue_number || data.pr_number, chatJid, groupFolder: group.folder }, 'Enqueued GitHub container task');
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

4. **Ask questions if ANYTHING is unclear**:
   - If the issue is vague, ambiguous, or you need clarification — post your questions as a comment on issue #${data.issue_number} using github_comment
   - Prefer asking over guessing. It's better to ask a clarifying question than to build the wrong thing.
   - After posting questions, **stop and exit**. A new container will spawn when the user replies.

5. **Assess the task complexity**:

   **SMALL/MEDIUM TASKS** — implement directly without waiting for approval:
   - Bug fixes (even if touching multiple files, as long as the fix is clear)
   - Single-file or few-file feature additions
   - Documentation updates (README, comments, docstrings, docs/)
   - Config changes (updating values, adding env vars, updating dependencies)
   - Adding functions, methods, or utilities with clear requirements
   - Dependency version bumps or package updates
   - Test additions or test fixes
   - Refactoring within a single module/component
   - UI tweaks and styling changes
   - Error handling improvements
   - Performance optimizations with obvious approaches
   - Code cleanup and linting fixes

   **General rule**: If the issue description is clear and the implementation approach is straightforward (even if it touches 3-5 files), proceed directly. Post a brief comment explaining what you'll do, then **implement it immediately** — create a branch, make changes, commit, open a PR, merge it, and comment with the result.

   **LARGE TASKS** — post a plan and wait for approval:
   - Major architectural changes or redesigns
   - New features with ambiguous requirements or multiple possible approaches
   - Changes affecting core system behavior in non-obvious ways
   - Database schema migrations or data model changes
   - Security-sensitive changes (auth, permissions, encryption)
   - Breaking API changes
   - Large refactors spanning many modules

   **General rule**: Only ask for approval if the requirements are unclear, there are multiple reasonable approaches to consider, or the changes are risky/sensitive. When in doubt, lean toward implementing directly.

   If the task is large: post your plan as a comment on issue #${data.issue_number} using github_comment, be specific about what files you'll change and why, then **stop and wait**. A new container will be spawned when the user approves.

**Important**:
- The repository persists in /workspace/group/ across runs - don't clone unnecessarily
- For small/medium tasks, implement end-to-end in one go (branch → code → commit → PR → merge)
- For large tasks, focus on creating a thorough plan and don't implement anything
- When in doubt about size, lean toward implementing directly — the user prefers fewer approval requests
- When in doubt about requirements (not approach), ASK. Questions about what to build are valuable, but if the "what" is clear, proceed with confidence even if there are implementation details to figure out.

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

6. **Merge the pull request** using github_merge_pr:
   - Use the PR number from step 5
   - Squash merge (default) is preferred
   - The plan was already human-approved, so no separate review is needed

7. **Comment on the issue** with the PR link and confirmation that it's been merged

**Important**:
- The repository persists in /workspace/group/${data.repo_name}/ - use it directly
- Follow the plan you created earlier
- Write clean, tested code
- After creating the PR, merge it immediately — your plan was already approved

Begin by navigating to the repository and reviewing your approved plan.`;
}

/**
 * Build prompt for question_answered event (user replied to agent's questions)
 */
function buildQuestionAnsweredPrompt(data: any): string {
  return `A user has replied to your questions on GitHub issue #${data.issue_number}.

**Repository**: ${data.repo_owner}/${data.repo_name}
**Issue #${data.issue_number}**: ${data.title}
**URL**: ${data.issue_url}

**Latest comment from ${data.commenter}**:
${data.comment_body}

---

**WORKSPACE**: The repository is in \`/workspace/group/${data.repo_name}/\`

**YOUR WORKFLOW**:

1. **Read the full comment history** using github_get_comments on issue #${data.issue_number} to understand the full context

2. **Check the repository**:
   - \`cd /workspace/group/${data.repo_name}\` and \`git pull\`
   - If it doesn't exist, clone it first

3. **Continue planning** based on the user's answers:
   - If you now have enough clarity, post your implementation plan as a comment
   - If you still have questions, post them as a comment and exit

4. **Stop and wait** - DO NOT implement yet.

**Important**:
- Read ALL comments, not just the latest — you need the full conversation context
- If the answer resolves your questions, proceed to post a complete plan
- If you need more clarification, ask specific follow-up questions and exit
- Don't implement anything in this phase - just plan

Begin by reading the full comment history.`;
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

  logger.info(
    {
      issue: issue.number,
      repo: `${repository.owner.login}/${repository.name}`,
      title: issue.title,
    },
    'Issue assigned to bot - spawning container',
  );

  // Spawn a container to handle this issue assignment
  spawnGitHubContainer('issue_assigned', {
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

  // Check if the comment is from the bot itself (ignore)
  const botUsername = process.env.GITHUB_BOT_USERNAME || 'bagel-bot';
  if (comment.user.login === botUsername) {
    logger.debug('Ignoring comment from bot itself');
    return;
  }

  // Check if bot is assigned to this issue
  const isAssignedToBot = issue.assignees.some((a) => a.login === botUsername);
  if (!isAssignedToBot) {
    logger.debug({ issue: issue.number }, 'Bot not assigned to this issue');
    return;
  }

  // Check for approval keywords in the comment
  const approvalKeywords = ['approved', 'lgtm', 'go ahead', 'looks good', 'ship it', '🚀'];
  const commentLower = comment.body.toLowerCase();
  const isApproval = approvalKeywords.some((keyword) => commentLower.includes(keyword));

  // React with eyes to acknowledge the comment, capture reaction ID for swap
  let eyesReactionId: number | undefined;
  try {
    eyesReactionId = await reactToComment(repository.owner.login, repository.name, comment.id, 'eyes');
  } catch (err) {
    logger.warn({ err, commentId: comment.id }, 'Failed to add eyes reaction to comment');
  }

  const emojiSwap: EmojiSwap | undefined = eyesReactionId !== undefined
    ? { owner: repository.owner.login, repo: repository.name, commentId: comment.id, eyesReactionId }
    : undefined;

  if (isApproval) {
    logger.info(
      {
        issue: issue.number,
        approver: comment.user.login,
      },
      'Approval detected in comment - spawning container to implement',
    );

    // Spawn a container to handle the approved implementation
    spawnGitHubContainer('plan_approved', {
      repo_owner: repository.owner.login,
      repo_name: repository.name,
      issue_number: issue.number,
      title: issue.title,
      issue_url: issue.html_url,
      approver: comment.user.login,
    }, emojiSwap);
  } else {
    // Non-approval comment on an assigned issue — user may be answering agent's questions
    logger.info(
      {
        issue: issue.number,
        commenter: comment.user.login,
      },
      'User comment on assigned issue - spawning question_answered container',
    );

    spawnGitHubContainer('question_answered', {
      repo_owner: repository.owner.login,
      repo_name: repository.name,
      issue_number: issue.number,
      title: issue.title,
      issue_url: issue.html_url,
      commenter: comment.user.login,
      comment_body: comment.body,
    }, emojiSwap);
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
export function startWebhookServer(port: number = 3000, groupQueue?: GroupQueue) {
  if (groupQueue) {
    queue = groupQueue;
  }
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
