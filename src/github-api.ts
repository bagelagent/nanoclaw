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
  const response = await client.pulls.merge({
    owner,
    repo,
    pull_number: pullNumber,
    merge_method: mergeMethod,
  });
  logger.info({ owner, repo, pullNumber, mergeMethod }, 'Merged pull request');
  return {
    sha: response.data.sha,
    merged: response.data.merged,
    message: response.data.message,
  };
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
