import { DocsImage } from "@/components/docs";
import { InstallSection, WebUiSection } from "@/components/install-section";
import { TopicFlowDiagram } from "@/components/topic-flow-diagram";
import defaultMdxComponents from "fumadocs-ui/mdx";
import type { MDXComponents } from "mdx/types";

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    img: DocsImage,
    InstallSection,
    WebUiSection,
    TopicFlowDiagram,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
