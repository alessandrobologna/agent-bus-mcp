import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";
import { Provider } from "@/components/provider";
import { gitConfig } from "@/lib/shared";
import "./global.css";

const sans = Manrope({
  subsets: ["latin"],
  variable: "--font-sans",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: {
    default: "Agent Bus MCP Docs",
    template: "%s | Agent Bus MCP Docs",
  },
  description: "Local, durable coordination docs for Agent Bus MCP.",
  metadataBase: new URL(`https://${gitConfig.user}.github.io`),
};

export default function Layout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`} suppressHydrationWarning>
      <body className="min-h-screen bg-fd-background text-fd-foreground antialiased">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
