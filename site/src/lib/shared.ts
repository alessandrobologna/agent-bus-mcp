import { resolveBasePath } from "../../base-path.shared.js";

export const appName = "Agent Bus MCP";
export const docsRoute = "/docs";
export const docsImageRoute = "/og/docs";
export const docsContentRoute = "/llms.mdx/docs";

export const gitConfig = {
  user: "alessandrobologna",
  repo: "agent-bus-mcp",
  branch: "main",
};

export const basePath = resolveBasePath({
  explicit: process.env.NEXT_PUBLIC_BASE_PATH,
});

export function withBasePath(path: string) {
  if (!path.startsWith("/") || !basePath || path.startsWith(`${basePath}/`) || path === basePath) {
    return path;
  }
  return `${basePath}${path}`;
}

export function docsHref(path = "") {
  if (!path) {
    return `${docsRoute}/`;
  }
  return `${docsRoute}/${path.replace(/^\/+/, "")}/`;
}
