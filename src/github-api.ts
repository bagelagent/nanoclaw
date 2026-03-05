import { Octokit } from '@octokit/rest';

import { logger } from './logger.js';

let octokit: Octokit | null = null;

export function initGitHubClient(token: string): void {
  octokit = new Octokit({ auth: token });
  logger.info('GitHub API client initialized');
}

export function getGitHubClient(): Octokit {
  if (!octokit) {
    throw new Error('GitHub client not initialized. Set GITHUB_TOKEN in .env');
  }
  return octokit;
}

export interface GitHubIssue {
  number: number;
  title: string;
  body: string | null;
  html_url: string;
  state: string;
  labels: Array<{ name: string }>;
  assignee: {
    login: string;
  } | null;
  assignees: Array<{ login: string }>;
  created_at: string;
  updated_at: string;
}

export interface GitHubPR {
  number: number;
  html_url: string;
  title: string;
}

export async function fetchIssue(
  owner: string,
  repo: string,
  issueNumber: number,
): Promise<GitHubIssue> {
  const client = getGitHubClient();
  const response = await client.issues.get({
    owner,
    repo,
    issue_number: issueNumber,
  });
  return response.data as GitHubIssue;
}

export async function createPullRequest(
  owner: string,
  repo: string,
  title: string,
  body: string,
  head: string,
  base: string = 'main',
): Promise<GitHubPR> {
  const client = getGitHubClient();
  const response = await client.pulls.create({
    owner,
    repo,
    title,
    body,
    head,
    base,
  });
  return {
    number: response.data.number,
    html_url: response.data.html_url,
    title: response.data.title,
  };
}

export async function commentOnIssue(
  owner: string,
  repo: string,
  issueNumber: number,
  body: string,
): Promise<void> {
  const client = getGitHubClient();
  await client.issues.createComment({
    owner,
    repo,
    issue_number: issueNumber,
    body,
  });
  logger.info({ owner, repo, issueNumber }, 'Commented on issue');
}

export async function getDefaultBranch(
  owner: string,
  repo: string,
): Promise<string> {
  const client = getGitHubClient();
  const response = await client.repos.get({
    owner,
    repo,
  });
  return response.data.default_branch;
}

export interface GitHubComment {
  id: number;
  user: {
    login: string;
  };
  body: string;
  created_at: string;
  updated_at: string;
}

export interface GitHubMergeResult {
  sha: string;
  merged: boolean;
  message: string;
}

export async function mergePullRequest(
  owner: string,
  repo: string,
  pullNumber: number,
  mergeMethod: 'merge' | 'squash' | 'rebase' = 'squash',
): Promise<GitHubMergeResult> {
  const client = getGitHubClient();

  // GitHub returns 405 "not mergeable" briefly after PR creation while it
  // computes mergeability. Retry a few times with backoff to handle this.
  const maxRetries = 4;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await client.pulls.merge({
        owner,
        repo,
        pull_number: pullNumber,
        merge_method: mergeMethod,
      });
      logger.info(
        { owner, repo, pullNumber, mergeMethod },
        'Merged pull request',
      );
      return {
        sha: response.data.sha,
        merged: response.data.merged,
        message: response.data.message,
      };
    } catch (err: any) {
      const is405 = err?.status === 405;
      if (is405 && attempt < maxRetries) {
        const delay = (attempt + 1) * 2000; // 2s, 4s, 6s, 8s
        logger.warn(
          { owner, repo, pullNumber, attempt: attempt + 1, delay },
          'PR not mergeable yet, retrying...',
        );
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw err;
    }
  }

  // Unreachable, but TypeScript needs it
  throw new Error('Merge failed after retries');
}

export async function reactToComment(
  owner: string,
  repo: string,
  commentId: number,
  reaction:
    | '+1'
    | '-1'
    | 'laugh'
    | 'confused'
    | 'heart'
    | 'hooray'
    | 'rocket'
    | 'eyes',
): Promise<number> {
  const client = getGitHubClient();
  const response = await client.reactions.createForIssueComment({
    owner,
    repo,
    comment_id: commentId,
    content: reaction,
  });
  logger.info({ owner, repo, commentId, reaction }, 'Reacted to comment');
  return response.data.id;
}

export async function deleteReaction(
  owner: string,
  repo: string,
  commentId: number,
  reactionId: number,
): Promise<void> {
  const client = getGitHubClient();
  await client.reactions.deleteForIssueComment({
    owner,
    repo,
    comment_id: commentId,
    reaction_id: reactionId,
  });
  logger.info(
    { owner, repo, commentId, reactionId },
    'Deleted reaction from comment',
  );
}

export async function reopenIssue(
  owner: string,
  repo: string,
  issueNumber: number,
): Promise<void> {
  const client = getGitHubClient();
  await client.issues.update({
    owner,
    repo,
    issue_number: issueNumber,
    state: 'open',
  });
  logger.info({ owner, repo, issueNumber }, 'Reopened issue');
}

export async function getIssueComments(
  owner: string,
  repo: string,
  issueNumber: number,
): Promise<GitHubComment[]> {
  const client = getGitHubClient();
  const response = await client.issues.listComments({
    owner,
    repo,
    issue_number: issueNumber,
  });
  return response.data as GitHubComment[];
}
