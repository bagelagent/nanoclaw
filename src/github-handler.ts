import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

import {
  commentOnIssue,
  createPullRequest,
  fetchIssue,
  getDefaultBranch,
  getIssueComments,
  mergePullRequest,
} from './github-api.js';
import { logger } from './logger.js';

// Host path for actual file operations (process.cwd() = project root)
const GITHUB_WORK_DIR = path.join(process.cwd(), 'github-work');

// Container path prefix — containers see the project root at /workspace/project/
const CONTAINER_PROJECT_PREFIX = '/workspace/project';

/**
 * Translate a container path to a host path for git/fs operations.
 * Container sends paths like /workspace/project/github-work/owner/repo
 * or /workspace/group/repo — both need to map to host filesystem paths.
 */
function containerToHostPath(p: string, sourceGroup?: string): string {
  if (p.startsWith(CONTAINER_PROJECT_PREFIX + '/')) {
    return path.join(process.cwd(), p.slice(CONTAINER_PROJECT_PREFIX.length));
  }
  if (p.startsWith('/workspace/group/') && sourceGroup) {
    const relPath = p.slice('/workspace/group/'.length);
    return path.join(process.cwd(), 'groups', sourceGroup, relPath);
  }
  return p;
}

/**
 * Translate a host path to a container path for return values.
 * Host clones to <project-root>/github-work/owner/repo,
 * container needs /workspace/project/github-work/owner/repo.
 */
function hostToContainerPath(p: string): string {
  const cwd = process.cwd();
  if (p.startsWith(cwd)) {
    return CONTAINER_PROJECT_PREFIX + p.slice(cwd.length);
  }
  return p;
}

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
 * Configure git credential helper for a cloned repo so pull/push work
 */
function configureRepoAuth(repoPath: string, token: string): void {
  // Use a store-based credential helper that injects the token for github.com
  execSync(
    `git config credential.helper '!f() { echo "username=x-access-token"; echo "password=${token}"; }; f'`,
    { cwd: repoPath, stdio: 'pipe' },
  );
  // Set git identity so commits are attributed to bagelagent
  execSync(`git config user.name 'bagelagent'`, {
    cwd: repoPath,
    stdio: 'pipe',
  });
  execSync(`git config user.email 'bagel.agent@yahoo.com'`, {
    cwd: repoPath,
    stdio: 'pipe',
  });
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
    // Ensure auth is configured (token may have changed)
    const token = process.env.GITHUB_TOKEN;
    if (token) {
      configureRepoAuth(repoPath, token);
    }
    try {
      execSync('git pull', { cwd: repoPath, stdio: 'pipe' });
      return { path: hostToContainerPath(repoPath) };
    } catch (err) {
      logger.warn({ err, owner, repo }, 'Failed to pull, will re-clone');
      // Fall through to clone
    }
  }

  // Clone repository
  fs.mkdirSync(path.dirname(repoPath), { recursive: true });

  // Inject token into clone URL for authentication
  const token = process.env.GITHUB_TOKEN;
  const cloneUrl = token
    ? `https://x-access-token:${token}@github.com/${owner}/${repo}.git`
    : `https://github.com/${owner}/${repo}.git`;

  logger.info({ owner, repo }, 'Cloning repository');

  try {
    execSync(`git clone "${cloneUrl}" "${repoPath}"`, {
      stdio: 'pipe',
      timeout: 120000, // 2 minute timeout
    });

    // Set the clean remote URL (without token) so it's not leaked in logs/config
    // Auth for future operations uses the credential helper below
    execSync(
      `git remote set-url origin "https://github.com/${owner}/${repo}.git"`,
      {
        cwd: repoPath,
        stdio: 'pipe',
      },
    );

    // Configure token-based credential helper for this repo
    if (token) {
      configureRepoAuth(repoPath, token);
    }

    logger.info({ owner, repo, repoPath }, 'Repository cloned successfully');
    return { path: hostToContainerPath(repoPath) };
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
  sourceGroup?: string,
): Promise<void> {
  // Translate container path to host path if needed
  repoPath = containerToHostPath(repoPath, sourceGroup);

  // Ensure auth is configured (repo may have been cloned by the agent via bash)
  const token = process.env.GITHUB_TOKEN;
  if (token) {
    configureRepoAuth(repoPath, token);
  }

  try {
    // Fetch latest from remote
    execSync('git fetch origin', { cwd: repoPath, stdio: 'pipe' });

    // Determine base branch if not specified
    if (!baseBranch) {
      const defaultBranch = execSync(
        'git symbolic-ref refs/remotes/origin/HEAD',
        {
          cwd: repoPath,
          encoding: 'utf-8',
        },
      )
        .trim()
        .replace('refs/remotes/origin/', '');
      baseBranch = defaultBranch;
    }

    // Checkout base branch
    execSync(`git checkout "${baseBranch}"`, { cwd: repoPath, stdio: 'pipe' });
    execSync('git pull', { cwd: repoPath, stdio: 'pipe' });

    // Create and checkout new branch
    execSync(`git checkout -b "${branchName}"`, {
      cwd: repoPath,
      stdio: 'pipe',
    });

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
  sourceGroup?: string,
): Promise<void> {
  // Translate container path to host path if needed
  repoPath = containerToHostPath(repoPath, sourceGroup);

  // Ensure auth is configured for push
  const token = process.env.GITHUB_TOKEN;
  if (token) {
    configureRepoAuth(repoPath, token);
  }

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
 * Update a branch by merging the latest from a base branch (e.g. main).
 * Returns status: 'updated' | 'up_to_date' | 'conflict'
 * On conflict, leaves conflict markers in files for the agent to resolve.
 * On other failures, aborts the merge and throws.
 */
export async function updateBranch(
  repoPath: string,
  baseBranch?: string,
  sourceGroup?: string,
): Promise<{
  status: 'updated' | 'up_to_date' | 'conflict';
  conflicted_files?: string[];
}> {
  repoPath = containerToHostPath(repoPath, sourceGroup);

  const token = process.env.GITHUB_TOKEN;
  if (token) {
    configureRepoAuth(repoPath, token);
  }

  try {
    execSync('git fetch origin', { cwd: repoPath, stdio: 'pipe' });
  } catch (err) {
    throw new Error(
      `Failed to fetch: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  // Determine base branch if not specified
  if (!baseBranch) {
    try {
      baseBranch = execSync('git symbolic-ref refs/remotes/origin/HEAD', {
        cwd: repoPath,
        encoding: 'utf-8',
      })
        .trim()
        .replace('refs/remotes/origin/', '');
    } catch {
      baseBranch = 'main';
    }
  }

  try {
    const output = execSync(`git merge origin/${baseBranch} --no-edit`, {
      cwd: repoPath,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Check for "Already up to date"
    if (output.includes('Already up to date')) {
      logger.info({ repoPath, baseBranch }, 'Branch already up to date');
      return { status: 'up_to_date' };
    }

    // Clean merge — push
    execSync('git push', { cwd: repoPath, stdio: 'pipe', timeout: 60000 });
    logger.info({ repoPath, baseBranch }, 'Branch updated and pushed');
    return { status: 'updated' };
  } catch (err) {
    // Check if this is a merge conflict
    try {
      const statusOutput = execSync('git status --porcelain', {
        cwd: repoPath,
        encoding: 'utf-8',
      });

      // Lines starting with "UU", "AA", "DD", etc. indicate conflicts
      const conflictedFiles = statusOutput
        .split('\n')
        .filter((line) => /^(UU|AA|DD|AU|UA|DU|UD) /.test(line))
        .map((line) => line.slice(3).trim());

      if (conflictedFiles.length > 0) {
        logger.info(
          { repoPath, baseBranch, conflictedFiles },
          'Merge conflicts detected',
        );
        return { status: 'conflict', conflicted_files: conflictedFiles };
      }
    } catch {
      // Couldn't check status, fall through to abort
    }

    // Not a conflict — abort the merge and throw
    try {
      execSync('git merge --abort', { cwd: repoPath, stdio: 'pipe' });
    } catch {
      // merge --abort might fail if there's no merge in progress
    }
    throw new Error(
      `Merge failed: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/**
 * Handle GitHub IPC requests
 */
export async function handleGitHubIpc(
  data: GitHubIpcRequest,
  sourceGroup?: string,
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
        const { owner, repo } = data as unknown as {
          owner: string;
          repo: string;
        };

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

        await createWorkBranch(
          repo_path,
          branch_name,
          base_branch,
          sourceGroup,
        );

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

        await commitAndPush(
          repo_path,
          branch_name,
          commit_message,
          sourceGroup,
        );

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

        const pr = await createPullRequest(
          owner,
          repo,
          title,
          body,
          head,
          targetBranch,
        );

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

      case 'github_merge_pr': {
        const { owner, repo, pull_number, merge_method } = data as unknown as {
          owner: string;
          repo: string;
          pull_number: number;
          merge_method?: 'merge' | 'squash' | 'rebase';
        };

        const mergeResult = await mergePullRequest(
          owner,
          repo,
          pull_number,
          merge_method,
        );

        return {
          status: 'success',
          data: mergeResult,
        };
      }

      case 'github_update_branch': {
        const { repo_path, base_branch } = data as unknown as {
          repo_path: string;
          base_branch?: string;
        };

        const updateResult = await updateBranch(
          repo_path,
          base_branch,
          sourceGroup,
        );

        return {
          status: 'success',
          data: updateResult,
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
