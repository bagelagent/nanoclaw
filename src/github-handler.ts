import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

import {
  commentOnIssue,
  createPullRequest,
  fetchIssue,
  getDefaultBranch,
  getIssueComments,
} from './github-api.js';
import { logger } from './logger.js';

const GITHUB_WORK_DIR = '/workspace/project/github-work';

export interface GitHubIpcRequest {
  type: string;
  requestId?: string;
  [key: string]: unknown;
}

export interface GitHubIpcReply {
  status: 'success' | 'error';
  data?: unknown;
  error?: string;
}

/**
 * Clone a GitHub repository to the workspace
 */
export async function cloneRepository(
  owner: string,
  repo: string,
): Promise<{ path: string }> {
  const repoPath = path.join(GITHUB_WORK_DIR, owner, repo);

  // Check if already cloned
  if (fs.existsSync(path.join(repoPath, '.git'))) {
    logger.info({ owner, repo }, 'Repository already cloned, pulling latest');
    try {
      execSync('git pull', { cwd: repoPath, stdio: 'pipe' });
      return { path: repoPath };
    } catch (err) {
      logger.warn({ err, owner, repo }, 'Failed to pull, will re-clone');
      // Fall through to clone
    }
  }

  // Clone repository
  fs.mkdirSync(path.dirname(repoPath), { recursive: true });

  // Use global git config for authentication
  const cloneUrl = `https://github.com/${owner}/${repo}.git`;

  logger.info({ owner, repo, cloneUrl }, 'Cloning repository');

  try {
    execSync(`git clone "${cloneUrl}" "${repoPath}"`, {
      stdio: 'pipe',
      timeout: 120000, // 2 minute timeout
    });
    logger.info({ owner, repo, repoPath }, 'Repository cloned successfully');
    return { path: repoPath };
  } catch (err) {
    logger.error({ err, owner, repo }, 'Failed to clone repository');
    throw new Error(
      `Failed to clone repository: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/**
 * Create a new branch for working on an issue
 */
export async function createWorkBranch(
  repoPath: string,
  branchName: string,
  baseBranch?: string,
): Promise<void> {
  try {
    // Fetch latest from remote
    execSync('git fetch origin', { cwd: repoPath, stdio: 'pipe' });

    // Determine base branch if not specified
    if (!baseBranch) {
      const defaultBranch = execSync('git symbolic-ref refs/remotes/origin/HEAD', {
        cwd: repoPath,
        encoding: 'utf-8',
      })
        .trim()
        .replace('refs/remotes/origin/', '');
      baseBranch = defaultBranch;
    }

    // Checkout base branch
    execSync(`git checkout "${baseBranch}"`, { cwd: repoPath, stdio: 'pipe' });
    execSync('git pull', { cwd: repoPath, stdio: 'pipe' });

    // Create and checkout new branch
    execSync(`git checkout -b "${branchName}"`, { cwd: repoPath, stdio: 'pipe' });

    logger.info({ repoPath, branchName, baseBranch }, 'Created work branch');
  } catch (err) {
    logger.error({ err, repoPath, branchName }, 'Failed to create work branch');
    throw new Error(
      `Failed to create branch: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/**
 * Commit and push changes to a branch
 */
export async function commitAndPush(
  repoPath: string,
  branchName: string,
  commitMessage: string,
): Promise<void> {
  try {
    // Stage all changes
    execSync('git add -A', { cwd: repoPath, stdio: 'pipe' });

    // Check if there are changes to commit
    try {
      execSync('git diff --cached --quiet', { cwd: repoPath, stdio: 'pipe' });
      // No changes
      logger.info({ repoPath }, 'No changes to commit');
      return;
    } catch {
      // Changes exist, continue with commit
    }

    // Commit
    execSync(`git commit -m "${commitMessage.replace(/"/g, '\\"')}"`, {
      cwd: repoPath,
      stdio: 'pipe',
    });

    // Push
    execSync(`git push -u origin "${branchName}"`, {
      cwd: repoPath,
      stdio: 'pipe',
      timeout: 60000, // 1 minute timeout
    });

    logger.info({ repoPath, branchName }, 'Committed and pushed changes');
  } catch (err) {
    logger.error({ err, repoPath, branchName }, 'Failed to commit and push');
    throw new Error(
      `Failed to commit and push: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/**
 * Handle GitHub IPC requests
 */
export async function handleGitHubIpc(
  data: GitHubIpcRequest,
): Promise<GitHubIpcReply> {
  try {
    switch (data.type) {
      case 'github_fetch_issue': {
        const { owner, repo, issue_number } = data as unknown as {
          owner: string;
          repo: string;
          issue_number: number;
        };

        const issue = await fetchIssue(owner, repo, issue_number);

        return {
          status: 'success',
          data: issue,
        };
      }

      case 'github_clone_repo': {
        const { owner, repo } = data as unknown as { owner: string; repo: string };

        const result = await cloneRepository(owner, repo);

        return {
          status: 'success',
          data: result,
        };
      }

      case 'github_create_branch': {
        const { repo_path, branch_name, base_branch } = data as unknown as {
          repo_path: string;
          branch_name: string;
          base_branch?: string;
        };

        await createWorkBranch(repo_path, branch_name, base_branch);

        return {
          status: 'success',
          data: { branch: branch_name },
        };
      }

      case 'github_commit_push': {
        const { repo_path, branch_name, commit_message } = data as unknown as {
          repo_path: string;
          branch_name: string;
          commit_message: string;
        };

        await commitAndPush(repo_path, branch_name, commit_message);

        return {
          status: 'success',
        };
      }

      case 'github_create_pr': {
        const { owner, repo, title, body, head, base } = data as unknown as {
          owner: string;
          repo: string;
          title: string;
          body: string;
          head: string;
          base?: string;
        };

        // Get default branch if not specified
        const targetBranch = base || (await getDefaultBranch(owner, repo));

        const pr = await createPullRequest(owner, repo, title, body, head, targetBranch);

        return {
          status: 'success',
          data: pr,
        };
      }

      case 'github_comment': {
        const { owner, repo, issue_number, body } = data as unknown as {
          owner: string;
          repo: string;
          issue_number: number;
          body: string;
        };

        await commentOnIssue(owner, repo, issue_number, body);

        return {
          status: 'success',
        };
      }

      case 'github_get_comments': {
        const { owner, repo, issue_number } = data as unknown as {
          owner: string;
          repo: string;
          issue_number: number;
        };

        const comments = await getIssueComments(owner, repo, issue_number);

        return {
          status: 'success',
          data: comments,
        };
      }

      default:
        return {
          status: 'error',
          error: `Unknown GitHub IPC type: ${data.type}`,
        };
    }
  } catch (err) {
    logger.error({ err, type: data.type }, 'GitHub IPC handler error');
    return {
      status: 'error',
      error: err instanceof Error ? err.message : String(err),
    };
  }
}
