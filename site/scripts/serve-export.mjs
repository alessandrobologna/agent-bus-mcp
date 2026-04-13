import { createServer } from "node:http";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const scriptDir = fileURLToPath(new URL("..", import.meta.url));
const outDir = join(scriptDir, "out");
const port = Number.parseInt(process.env.PORT ?? "3000", 10);

function normalizeBasePath(value) {
  if (!value || value === "/") return "";
  const trimmed = value.replace(/\/+$/, "");
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

function inferBasePath() {
  if (process.env.NEXT_PUBLIC_BASE_PATH !== undefined) {
    return normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);
  }

  if (process.env.GITHUB_ACTIONS === "true" && process.env.GITHUB_REPOSITORY) {
    const repo = process.env.GITHUB_REPOSITORY.split("/")[1];
    if (repo) return `/${repo}`;
  }

  const indexPath = join(outDir, "index.html");
  if (!existsSync(indexPath)) return "";

  const html = readFileSync(indexPath, "utf8");
  const match = html.match(/(?:href|src)="(\/[^"]*?)\/_next\//);
  return normalizeBasePath(match?.[1] ?? "");
}

const basePath = inferBasePath();

function getContentType(filePath, requestPath) {
  const extension = extname(filePath).toLowerCase();
  switch (extension) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".css":
      return "text/css; charset=utf-8";
    case ".js":
      return "text/javascript; charset=utf-8";
    case ".json":
      return "application/json; charset=utf-8";
    case ".txt":
      return "text/plain; charset=utf-8";
    case ".svg":
      return "image/svg+xml";
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".webp":
      return "image/webp";
    case ".ico":
      return "image/x-icon";
    case ".woff":
      return "font/woff";
    case ".woff2":
      return "font/woff2";
    default:
      if (requestPath.startsWith("/api/")) return "application/json; charset=utf-8";
      return "application/octet-stream";
  }
}

function resolveFilePath(pathname) {
  const relativePath = pathname.replace(/^\/+/, "");
  const normalized = normalize(relativePath);
  if (normalized.startsWith("..")) return null;

  const candidates = [];
  if (!normalized || normalized === ".") {
    candidates.push(join(outDir, "index.html"));
  } else {
    candidates.push(join(outDir, normalized));
    candidates.push(join(outDir, normalized, "index.html"));
  }

  for (const candidate of candidates) {
    if (!existsSync(candidate)) continue;
    const stats = statSync(candidate);
    if (stats.isFile()) return candidate;
  }

  return null;
}

function sendFile(res, filePath, statusCode, method, requestPath) {
  const headers = {
    "Content-Type": getContentType(filePath, requestPath),
    "Cache-Control": requestPath.includes("/_next/") ? "public, max-age=31536000, immutable" : "no-cache",
  };
  res.writeHead(statusCode, headers);
  if (method === "HEAD") {
    res.end();
    return;
  }
  createReadStream(filePath).pipe(res);
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  let pathname;
  try {
    pathname = decodeURIComponent(url.pathname);
  } catch {
    res.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Malformed request path");
    return;
  }

  if (basePath) {
    if (pathname === "/") {
      res.writeHead(307, { Location: `${basePath}/` });
      res.end();
      return;
    }

    if (pathname === basePath) pathname = "/";
    else if (pathname.startsWith(`${basePath}/`)) pathname = pathname.slice(basePath.length);
    else {
      const notFoundFile = resolveFilePath("/404.html");
      if (notFoundFile) sendFile(res, notFoundFile, 404, req.method ?? "GET", "/404.html");
      else {
        res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
        res.end("Not found");
      }
      return;
    }
  }

  const filePath = resolveFilePath(pathname);
  if (!filePath) {
    const notFoundFile = resolveFilePath("/404.html");
    if (notFoundFile) sendFile(res, notFoundFile, 404, req.method ?? "GET", "/404.html");
    else {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not found");
    }
    return;
  }

  sendFile(res, filePath, 200, req.method ?? "GET", pathname);
});

server.listen(port, "127.0.0.1", () => {
  const baseUrl = `http://127.0.0.1:${port}`;
  const previewUrl = basePath ? `${baseUrl}${basePath}/` : `${baseUrl}/`;
  console.log(`Serving static export from ${outDir}`);
  console.log(`Base path: ${basePath || "/"}`);
  console.log(`Preview URL: ${previewUrl}`);
});
