import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter } from "react-router-dom"
import { Toaster } from "sonner"

import { TooltipProvider } from "@/components/ui/tooltip"

import App from "./App"
import "./index.css"

document.documentElement.classList.add("dark")
document.body.classList.add("dark")

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <TooltipProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <Toaster position="top-right" richColors />
    </TooltipProvider>
  </StrictMode>,
)
