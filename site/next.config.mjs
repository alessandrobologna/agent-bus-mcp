import { readFileSync } from "node:fs";
import { createMDX } from "fumadocs-mdx/next";
import { resolveBasePath } from "./base-path.shared.js";

const withMDX = createMDX();

const basePath = resolveBasePath();
const pyproject = readFileSync(new URL("../pyproject.toml", import.meta.url), "utf8");
const versionMatch = pyproject.match(/^version = "([^"]+)"$/m);

if (!versionMatch) {
  throw new Error("Unable to read Agent Bus version from pyproject.toml");
}

const agentBusVersion = versionMatch[1];

/** @type {import("next").NextConfig} */
const config = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
  basePath,
  images: {
    unoptimized: true,
  },
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
    NEXT_PUBLIC_AGENT_BUS_PACKAGE: "agent-bus-mcp",
    NEXT_PUBLIC_AGENT_BUS_VERSION: agentBusVersion,
  },
};

export default withMDX(config);
