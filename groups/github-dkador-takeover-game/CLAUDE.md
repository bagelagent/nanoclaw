# GitHub: dkador/takeover-game

This is an auto-created group for handling GitHub webhooks for the repository **dkador/takeover-game**.

## Workspace

The repository is cloned to `/workspace/group/takeover-game/` and persists across container runs.

## Memory

Conversation history and context for this repository's issues and PRs are stored here.

## Test Environment

**IMPORTANT:** The development environment has full testing capabilities. You MUST run tests before and after making changes.

### Available Test Suites

1. **Jest Unit & Integration Tests**
   ```bash
   cd /workspace/group/takeover-game
   npm test                  # Run all Jest tests
   npm run test:watch        # Watch mode
   npm run test:coverage     # With coverage report (80% threshold)
   ```

2. **Playwright E2E Tests**
   ```bash
   cd /workspace/group/takeover-game
   npm run test:e2e          # Runs solo and multiplayer E2E tests
   ```
   - Includes multiplayer tests in `tests/multiplayer/`
   - Tests basic flow, chain formation, mergers, stock buying
   - Auto-starts both frontend and server

3. **Server Tests**
   ```bash
   cd /workspace/group/takeover-game/server
   npm test                  # Server-specific tests
   ```

4. **Rule Validation**
   ```bash
   cd /workspace/group/takeover-game
   npm run test:rules        # Static analysis of game rules
   ```

### Test Requirements

- **Before implementing any fix:** Run existing tests to confirm they catch the bug
- **After implementing a fix:** All tests must pass
- **For new features:** Add corresponding test coverage
- **Coverage threshold:** Maintain ≥80% coverage on all metrics

### Dependencies

Test dependencies are available after running `npm install`:
- Jest with jsdom environment
- Playwright with browsers (requires `npx playwright install chromium`)
- Supertest for server testing
- PostgreSQL test database (required for E2E and server tests)

**Note:** E2E tests and server integration tests require PostgreSQL to be running locally. If PostgreSQL is not available, you can still run Jest unit tests with:
```bash
npm test -- --testPathIgnorePatterns=server
```

### Test Setup

**Quick Setup (Recommended):**

Use the automated setup script to configure the entire E2E test environment:

```bash
cd /workspace/group/takeover-game
./scripts/setup-e2e.sh
```

This idempotent script will:
- Start PostgreSQL server (if not running)
- Install all npm dependencies (project + server)
- Install Playwright browsers (chromium)
- Create test database with correct permissions
- Run database migrations
- Verify the environment is ready

**Manual Setup (if needed):**

1. Start PostgreSQL:
   ```bash
   pg_ctl -D /home/node/pgdata -l /tmp/pg.log start
   ```

2. Install project dependencies:
   ```bash
   cd /workspace/group/takeover-game
   npm install
   ```

3. Install server dependencies:
   ```bash
   cd /workspace/group/takeover-game/server
   npm install
   ```

4. Install Playwright browsers:
   ```bash
   npx playwright install chromium
   ```

5. Set up test database:
   ```bash
   psql -U node -c "CREATE DATABASE takeover_test OWNER dkador;" postgres
   cd server
   DATABASE_URL="postgresql://dkador@localhost:5432/takeover_test" npx prisma migrate deploy
   ```
