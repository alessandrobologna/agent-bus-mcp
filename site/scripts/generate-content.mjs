import { spawnSync } from "node:child_process";

const candidates = [
  { command: process.platform === "win32" ? "py" : "python3", args: process.platform === "win32" ? ["-3"] : [] },
  { command: "python", args: [] },
];

for (const candidate of candidates) {
  const result = spawnSync(
    candidate.command,
    [...candidate.args, "../scripts/generate_fumadocs_site.py"],
    {
      cwd: new URL("..", import.meta.url),
      stdio: "inherit",
    }
  );

  if (result.error?.code === "ENOENT") {
    continue;
  }

  process.exit(result.status ?? 1);
}

console.error("Unable to find a Python interpreter. Tried python3, py -3, and python.");
process.exit(1);
