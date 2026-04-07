import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
  },
  webServer: {
    command:
      "pnpm build && rm -f .playwright-agent-bus.sqlite* && " +
      "uv run --project .. python ../tests/fixtures/seed_web_ui_db.py .playwright-agent-bus.sqlite && " +
      "uv run --project .. agent-bus serve --host 127.0.0.1 --port 4173 --db-path .playwright-agent-bus.sqlite",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
