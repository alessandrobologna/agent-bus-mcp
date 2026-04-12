import type { Metadata } from "next";
import { getMDXComponents } from "@/components/mdx";
import { getPageImage, getPageMarkdownUrl, getSourceRepoPath, source } from "@/lib/source";
import { gitConfig } from "@/lib/shared";
import type { InferPageType } from "fumadocs-core/source";
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
  MarkdownCopyButton,
  ViewOptionsPopover,
} from "fumadocs-ui/layouts/docs/page";

type DocsSourcePage = InferPageType<typeof source>;

export function findDocsPage(slug?: string[]) {
  if (!slug || slug.length === 0) {
    return source.getPages().find((page) => page.path === "index.mdx");
  }

  return source.getPage(slug);
}

export function renderDocsPage(page: DocsSourcePage) {
  const MDX = page.data.body;
  const markdownUrl = getPageMarkdownUrl(page).url;
  const sourceRepoPath = getSourceRepoPath(page);

  return (
    <DocsPage toc={page.data.toc} full={page.data.full}>
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription className="mb-0">{page.data.description}</DocsDescription>
      <div className="flex flex-row items-center gap-2 border-b pb-6">
        <MarkdownCopyButton markdownUrl={markdownUrl} />
        <ViewOptionsPopover
          markdownUrl={markdownUrl}
          githubUrl={`https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/${sourceRepoPath}`}
        />
      </div>
      <DocsBody>
        <MDX components={getMDXComponents()} />
      </DocsBody>
    </DocsPage>
  );
}

export function buildDocsMetadata(page: DocsSourcePage): Metadata {
  return {
    title: page.data.title,
    description: page.data.description,
    openGraph: {
      images: getPageImage(page).url,
    },
  };
}
