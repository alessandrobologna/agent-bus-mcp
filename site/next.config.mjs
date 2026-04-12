import { createMDX } from "fumadocs-mdx/next";

const withMDX = createMDX();

function normalizeBasePath(value) {
  if (!value || value === "/") return "";
  const trimmed = value.replace(/\/+$/, "");
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

function resolveBasePath() {
  const explicit = process.env.NEXT_PUBLIC_BASE_PATH;
  if (explicit !== undefined) return normalizeBasePath(explicit);

  if (process.env.GITHUB_ACTIONS === "true" && process.env.GITHUB_REPOSITORY) {
    const repo = process.env.GITHUB_REPOSITORY.split("/")[1];
    if (repo) return `/${repo}`;
  }

  return "";
}

const basePath = resolveBasePath();

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
  },
};

export default withMDX(config);
