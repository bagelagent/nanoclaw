/**
 * GitHub Issue Worker
 * Spawns agent to work on GitHub issues with plan-approve-implement workflow
 */

import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import { logger } from './logger.js';
import { runContainerAgent } from './container-runner.js';
import { updateGitHubAssignment, GitHubAssignment, getAllRegisteredGroups } from './db.js';
import { commentOnIssue } from './github-api.js';

/**
 * Start working on a GitHub issue assignment
 *
 * Workflow:
 * 1. Spawn agent in plan mode
 * 2. Agent explores code and creates plan
 * 3. Post plan as comment on issue
 * 4. Wait for user approval (handled by separate polling)
 * 5. After approval: implement and create PR
 */
export async function startWorkingOnIssue(assignment: GitHubAssignment): Promise<void> {
  const { id, repo_owner, repo_name, issue_number, title, description } = assignment;

  logger.info(
    { assignmentId: id, repo: `${repo_owner}/${repo_name}`, issue: issue_number },
    'Starting work on GitHub issue',
  );

  // Update status to in_progress
  updateGitHubAssignment(id, {
    status: 'in_progress',
    last_updated: new Date().toISOString(),
  });

  // Post initial comment on issue
  try {
    await commentOnIssue(
      repo_owner,
      repo_name,
      issue_number,
      '🤖 I\'ve been assigned to this issue and I\'m starting work on it now.\n\n' +
      'I\'ll explore the codebase, create a plan, and post it here for review before implementing.',
    );
  } catch (err) {
    logger.error({ err, assignmentId: id }, 'Failed to post initial comment');
  }

  // Build prompt for the agent
  const prompt = buildAgentPrompt(assignment);

  // Spawn agent to work on the issue
  // Use main group context since GitHub integration is main-only
  const registeredGroups = getAllRegisteredGroups();
  const mainGroup = Object.values(registeredGroups).find((g) => g.folder === 'main');

  if (!mainGroup) {
    throw new Error('Main group not found in registered groups');
  }

  const mainChatJid = `github-worker-${id}`; // Virtual chat ID for GitHub work
  const sessionId = `github-${id}-${Date.now()}`;

  try {
    const output = await runContainerAgent(
      mainGroup,
      {
        prompt,
        sessionId,
        groupFolder: mainGroup.folder,
        chatJid: mainChatJid,
        isMain: true,
      },
      () => {}, // No-op callback for process tracking
    );

    logger.info(
      { assignmentId: id, status: output.status },
      'Agent completed GitHub issue work',
    );

    // The agent will handle posting plan, waiting for approval, implementing, and creating PR
    // All through the MCP tools we've built

  } catch (err) {
    logger.error({ err, assignmentId: id }, 'Agent failed while working on GitHub issue');

    // Update assignment status to blocked
    updateGitHubAssignment(id, {
      status: 'blocked',
      last_updated: new Date().toISOString(),
      notes: `Error: ${err instanceof Error ? err.message : String(err)}`,
    });

    // Post error comment on issue
    try {
      await commentOnIssue(
        repo_owner,
        repo_name,
        issue_number,
        `❌ I encountered an error while working on this issue:\n\n\`\`\`\n${err instanceof Error ? err.message : String(err)}\n\`\`\`\n\nI've marked this as blocked. Please check the logs.`,
      );
    } catch (commentErr) {
      logger.error({ err: commentErr, assignmentId: id }, 'Failed to post error comment');
    }
  }
}

/**
 * Build the agent prompt for working on a GitHub issue
 */
function buildAgentPrompt(assignment: GitHubAssignment): string {
  const { repo_owner, repo_name, issue_number, title, description, issue_url } = assignment;

  return `You've been assigned to work on a GitHub issue. Here are the details:

**Repository**: ${repo_owner}/${repo_name}
**Issue #${issue_number}**: ${title}
**URL**: ${issue_url}

**Description**:
${description || '(No description provided)'}

---

**CRITICAL WORKFLOW - Follow these steps exactly**:

1. **Clone the repository**
   - Use: github_clone_repo with owner="${repo_owner}" and repo="${repo_name}"
   - This returns the local path to work with

2. **Explore the codebase thoroughly**
   - Read relevant files to understand the issue
   - Use Grep, Read, and Glob tools to navigate
   - Understand the project structure and patterns

3. **Design your solution (PLAN MODE)**
   - Use EnterPlanMode to create a detailed implementation plan
   - Your plan should include:
     * Problem analysis
     * Proposed solution approach
     * Files that will be modified
     * Testing strategy
     * Any risks or considerations
   - Use ExitPlanMode when your plan is complete

4. **Post the plan for approval**
   - Use github_comment to post your plan on issue #${issue_number}
   - Format it clearly with markdown
   - Include: what you'll change, why, and how

5. **Wait for user approval**
   - Use github_get_comments to check for new comments
   - Look for approval keywords: "approved", "lgtm", "go ahead", "looks good", "ship it"
   - If you see change requests, update your plan accordingly
   - DO NOT proceed to implementation until you have clear approval

6. **Implement the solution**
   - Create a new branch using github_create_branch (use format: "bagel/issue-${issue_number}-brief-description")
   - Make your code changes using Edit and Write tools
   - Follow the plan you created
   - Commit and push using github_commit_push

7. **Create a pull request**
   - Use github_create_pr with:
     * Clear title referencing the issue
     * Detailed description of changes
     * Link to the issue: "Fixes #${issue_number}"
   - The PR should be ready for review

8. **Update the issue**
   - Use github_comment to add a comment with the PR link
   - Include a summary of what was implemented

**Important reminders**:
- You MUST wait for approval before implementing - this is not optional
- Be thorough in exploration - understand before you plan
- Create a detailed plan - this is a real person reviewing it
- Test your changes if possible before creating the PR
- Write clear, professional commit messages and PR descriptions

Begin by cloning the repository and exploring it to understand the issue.`;
}
