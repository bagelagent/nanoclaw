/**
 * GitHub MCP Tools for NanoClaw Agent
 * Provides tools for working with GitHub issues and pull requests via IPC
 */

import { tool } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

const IPC_DIR = '/workspace/ipc';
const TASKS_DIR = path.join(IPC_DIR, 'tasks');

function writeIpcFile(dir: string, data: any): string {
  fs.mkdirSync(dir, { recursive: true });

  const requestId = crypto.randomUUID();
  const filename = `${requestId}.json`;
  const filepath = path.join(dir, filename);

  // Add requestId to payload for reply matching
  data.requestId = requestId;

  // Atomic write: temp file then rename
  const tempPath = `${filepath}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(data, null, 2));
  fs.renameSync(tempPath, filepath);

  return requestId;
}

async function waitForReply(requestId: string, timeoutMs: number): Promise<any> {
  const REPLIES_DIR = path.join('/workspace/ipc', 'replies');
  const replyPath = path.join(REPLIES_DIR, `${requestId}.json`);
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    try {
      if (fs.existsSync(replyPath)) {
        const reply = JSON.parse(fs.readFileSync(replyPath, 'utf-8'));
        fs.unlinkSync(replyPath); // Clean up
        return reply;
      }
    } catch (err) {
      // File might be mid-write, continue polling
    }
    await new Promise(resolve => setTimeout(resolve, 500)); // Poll every 500ms
  }

  throw new Error(`Timeout waiting for reply after ${timeoutMs}ms`);
}

export interface GitHubMcpContext {
  chatJid: string;
  groupFolder: string;
  isMain: boolean;
}

export function createGitHubMcpTools(ctx: GitHubMcpContext) {
  const { isMain, groupFolder } = ctx;

  // Only main group or GitHub groups can use GitHub tools
  const isGitHubGroup = groupFolder.startsWith('github-');
  if (!isMain && !isGitHubGroup) {
    return [];
  }

  return [
    tool(
      'github_fetch_issue',
      'Fetch details about a GitHub issue. Returns issue title, description, labels, and other metadata.',
      {
        owner: z.string().describe('Repository owner (e.g., "dkador")'),
        repo: z.string().describe('Repository name (e.g., "takeover-game")'),
        issue_number: z.number().describe('Issue number'),
      },
      async (args: { owner: string; repo: string; issue_number: number }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_fetch_issue',
          owner: args.owner,
          repo: args.repo,
          issue_number: args.issue_number,
        });

        try {
          const reply = await waitForReply(requestId, 30000);

          if (reply.status === 'success') {
            return {
              content: [
                {
                  type: 'text',
                  text: reply.data ? JSON.stringify(reply.data, null, 2) : 'Success (no data)',
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to fetch issue: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error fetching issue: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),

    tool(
      'github_clone_repo',
      'Clone a GitHub repository to the local workspace. Returns the path to the cloned repository.',
      {
        owner: z.string().describe('Repository owner'),
        repo: z.string().describe('Repository name'),
      },
      async (args: { owner: string; repo: string }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_clone_repo',
          owner: args.owner,
          repo: args.repo,
        });

        try {
          const reply = await waitForReply(requestId, 120000); // 2 min timeout for clone

          if (reply.status === 'success') {
            return {
              content: [
                {
                  type: 'text',
                  text: `Repository cloned to: ${reply.data.path}`,
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to clone repository: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error cloning repository: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),

    tool(
      'github_create_branch',
      'Create a new git branch in a cloned repository.',
      {
        repo_path: z.string().describe('Path to the cloned repository'),
        branch_name: z.string().describe('Name for the new branch (e.g., "bagel/issue-5-fix-bug")'),
        base_branch: z.string().optional().describe('Base branch to branch from (defaults to main/master)'),
      },
      async (args: { repo_path: string; branch_name: string; base_branch?: string }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_create_branch',
          repo_path: args.repo_path,
          branch_name: args.branch_name,
          base_branch: args.base_branch,
        });

        try {
          const reply = await waitForReply(requestId, 30000);

          if (reply.status === 'success') {
            return {
              content: [
                {
                  type: 'text',
                  text: `Branch created: ${args.branch_name}`,
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to create branch: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error creating branch: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),

    tool(
      'github_commit_push',
      'Commit all changes in the repository and push to remote.',
      {
        repo_path: z.string().describe('Path to the repository'),
        branch_name: z.string().describe('Branch name to push'),
        commit_message: z.string().describe('Commit message'),
      },
      async (args: { repo_path: string; branch_name: string; commit_message: string }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_commit_push',
          repo_path: args.repo_path,
          branch_name: args.branch_name,
          commit_message: args.commit_message,
        });

        try {
          const reply = await waitForReply(requestId, 60000); // 1 min timeout

          if (reply.status === 'success') {
            return {
              content: [
                {
                  type: 'text',
                  text: 'Changes committed and pushed successfully',
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to commit and push: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error committing and pushing: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),

    tool(
      'github_create_pr',
      'Create a pull request on GitHub.',
      {
        owner: z.string().describe('Repository owner'),
        repo: z.string().describe('Repository name'),
        title: z.string().describe('PR title'),
        body: z.string().describe('PR description/body'),
        head: z.string().describe('Source branch (e.g., "bagel/issue-5-fix")'),
        base: z.string().optional().describe('Target branch (defaults to main)'),
      },
      async (args: {
        owner: string;
        repo: string;
        title: string;
        body: string;
        head: string;
        base?: string;
      }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_create_pr',
          owner: args.owner,
          repo: args.repo,
          title: args.title,
          body: args.body,
          head: args.head,
          base: args.base,
        });

        try {
          const reply = await waitForReply(requestId, 30000);

          if (reply.status === 'success') {
            const pr = reply.data;
            return {
              content: [
                {
                  type: 'text',
                  text: `Pull request created!\n\nPR #${pr.number}: ${pr.title}\n${pr.html_url}`,
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to create PR: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error creating PR: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),

    tool(
      'github_comment',
      'Add a comment to a GitHub issue or pull request.',
      {
        owner: z.string().describe('Repository owner'),
        repo: z.string().describe('Repository name'),
        issue_number: z.number().describe('Issue or PR number'),
        body: z.string().describe('Comment text'),
      },
      async (args: { owner: string; repo: string; issue_number: number; body: string }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_comment',
          owner: args.owner,
          repo: args.repo,
          issue_number: args.issue_number,
          body: args.body,
        });

        try {
          const reply = await waitForReply(requestId, 30000);

          if (reply.status === 'success') {
            return {
              content: [
                {
                  type: 'text',
                  text: 'Comment posted successfully',
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to post comment: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error posting comment: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),

    tool(
      'github_merge_pr',
      'Merge a pull request on GitHub. Use after creating a PR to auto-merge it.',
      {
        owner: z.string().describe('Repository owner'),
        repo: z.string().describe('Repository name'),
        pull_number: z.number().describe('Pull request number'),
        merge_method: z.enum(['merge', 'squash', 'rebase']).optional().describe('Merge method (defaults to squash)'),
      },
      async (args: { owner: string; repo: string; pull_number: number; merge_method?: 'merge' | 'squash' | 'rebase' }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_merge_pr',
          owner: args.owner,
          repo: args.repo,
          pull_number: args.pull_number,
          merge_method: args.merge_method,
        });

        try {
          const reply = await waitForReply(requestId, 30000);

          if (reply.status === 'success') {
            const result = reply.data;
            return {
              content: [
                {
                  type: 'text',
                  text: `Pull request merged!\n\nSHA: ${result.sha}\nMerged: ${result.merged}\nMessage: ${result.message}`,
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to merge PR: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error merging PR: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),

    tool(
      'github_update_branch',
      'Update the current branch by merging the latest changes from the base branch (e.g. main). Use this when a merge or push fails because the branch is behind. If conflicts are returned, resolve them in the conflicted files, then commit and push.',
      {
        repo_path: z.string().describe('Path to the repository'),
        base_branch: z.string().optional().describe('Base branch to merge from (defaults to main/master)'),
      },
      async (args: { repo_path: string; base_branch?: string }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_update_branch',
          repo_path: args.repo_path,
          base_branch: args.base_branch,
        });

        try {
          const reply = await waitForReply(requestId, 60000); // 1 min timeout

          if (reply.status === 'success') {
            const result = reply.data;
            if (result.status === 'conflict') {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Merge conflicts detected. Conflicted files:\n${result.conflicted_files.map((f: string) => `- ${f}`).join('\n')}\n\nResolve the conflicts in these files (remove conflict markers), then use github_commit_push to commit and push.`,
                  },
                ],
              };
            }
            return {
              content: [
                {
                  type: 'text',
                  text: result.status === 'updated'
                    ? 'Branch updated successfully with latest changes from base branch and pushed.'
                    : 'Branch is already up to date with the base branch.',
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to update branch: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error updating branch: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),

    tool(
      'github_get_comments',
      'Get all comments on a GitHub issue or pull request. Use this to check for user approval after posting a plan.',
      {
        owner: z.string().describe('Repository owner'),
        repo: z.string().describe('Repository name'),
        issue_number: z.number().describe('Issue or PR number'),
      },
      async (args: { owner: string; repo: string; issue_number: number }) => {
        const requestId = writeIpcFile(TASKS_DIR, {
          type: 'github_get_comments',
          owner: args.owner,
          repo: args.repo,
          issue_number: args.issue_number,
        });

        try {
          const reply = await waitForReply(requestId, 30000);

          if (reply.status === 'success') {
            return {
              content: [
                {
                  type: 'text',
                  text: reply.data ? JSON.stringify(reply.data, null, 2) : 'No comments found',
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to get comments: ${reply.error}`,
                },
              ],
              isError: true,
            };
          }
        } catch (err) {
          return {
            content: [
              {
                type: 'text',
                text: `Timeout or error getting comments: ${err instanceof Error ? err.message : String(err)}`,
              },
            ],
            isError: true,
          };
        }
      },
    ),
  ];
}
