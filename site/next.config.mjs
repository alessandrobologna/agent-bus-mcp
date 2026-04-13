import { createMDX } from "fumadocs-mdx/next";
import { resolveBasePath } from "./base-path.shared.js";

const withMDX = createMDX();

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
