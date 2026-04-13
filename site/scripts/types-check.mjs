import fs from "node:fs";
import { spawnSync } from "node:child_process";

for (const path of [".next/types", ".next/dev/types"]) {
  fs.rmSync(new URL(`../${path}`, import.meta.url), { recursive: true, force: true });
}

for (const command of [
  ["pnpm", ["run", "generate:content"]],
  ["pnpm", ["exec", "fumadocs-mdx"]],
  ["pnpm", ["exec", "next", "typegen"]],
  ["pnpm", ["exec", "tsc", "--noEmit"]],
]) {
  const result = spawnSync(command[0], command[1], {
    cwd: new URL("..", import.meta.url),
    stdio: "inherit",
  });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
