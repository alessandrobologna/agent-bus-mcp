import { docs } from "collections/server";
import { type InferPageType, loader } from "fumadocs-core/source";
import { docsContentRoute, docsImageRoute, docsRoute, withBasePath } from "@/lib/shared";

export const source = loader({
  baseUrl: docsRoute,
  source: docs.toFumadocsSource(),
  plugins: [],
});

export function getPageImage(page: InferPageType<typeof source>) {
  const segments = [...page.slugs, "image.png"];

  return {
    segments,
    url: withBasePath(`${docsImageRoute}/${segments.join("/")}`),
  };
}

export function getPageMarkdownUrl(page: InferPageType<typeof source>) {
  const segments = [...page.slugs, "content.md"];

  return {
    segments,
    url: withBasePath(`${docsContentRoute}/${segments.join("/")}`),
  };
}

export function getSourceRepoPath(page: InferPageType<typeof source>) {
  if (page.path === "index.mdx") return "docs/README.md";
  if (page.path === "reference/implementation-spec.mdx") return "spec.md";
  if (page.path === "reference/changelog.mdx") return "CHANGELOG.md";
  if (page.path.endsWith("/index.mdx")) {
    return `docs/${page.path.replace(/\/index\.mdx$/, "/README.md")}`;
  }
  return `docs/${page.path.replace(/\.mdx$/, ".md")}`;
}

export async function getLLMText(page: InferPageType<typeof source>) {
  const processed = await page.data.getText("processed");

  return `# ${page.data.title} (${page.url})\n\n${processed}`;
}
