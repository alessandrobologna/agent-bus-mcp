import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { buildDocsMetadata, findDocsPage, renderDocsPage } from "@/lib/docs-page";

export default function DocsIndexPage() {
  const page = findDocsPage();
  if (!page) notFound();

  return renderDocsPage(page);
}

export function generateMetadata(): Metadata {
  const page = findDocsPage();
  if (!page) notFound();

  return buildDocsMetadata(page);
}
