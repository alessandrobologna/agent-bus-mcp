export const appName = "Agent Bus";
export const docsRoute = "/docs";
export const docsImageRoute = "/og/docs";
export const docsContentRoute = "/llms.mdx/docs";

export const gitConfig = {
  user: "alessandrobologna",
  repo: "agent-bus-mcp",
  branch: "main",
};

function normalizeBasePath(value: string | undefined) {
  if (!value || value === "/") return "";
  const trimmed = value.replace(/\/+$/, "");
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

export const basePath = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);

export function withBasePath(path: string) {
  if (!path.startsWith("/") || !basePath || path.startsWith(`${basePath}/`) || path === basePath) {
    return path;
  }
  return `${basePath}${path}`;
}
