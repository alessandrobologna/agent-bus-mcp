import { expect, test } from "@playwright/test"

test("opens topics in tabs and navigates from sidebar search", async ({ page }) => {
  await page.goto("/")

  await expect(page.getByText("Alpha review")).toBeVisible()
  await page.getByRole("button", { name: /Alpha review/i }).click()
  await expect(page.getByText("hello from alpha")).toBeVisible()

  await page.getByPlaceholder("Search").fill("handoff")
  const result = page.getByRole("button", { name: /Beta thread/i }).first()
  await expect(result).toBeVisible()
  await result.click()

  await expect(page).toHaveURL(/\/topics\/.+\?focus=/)
  await expect(page.locator("main").getByText("beta handoff summary")).toBeVisible()
})

test("reveals the thread-map overlay for long desktop topics while keeping the inspector rail", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 700 })
  await page.goto("/")

  await page.getByRole("button", { name: /Alpha review/i }).click()
  await page.locator("[data-ab-topic-thread-scroll-area='true']").evaluate((element) => {
    ;(element as HTMLElement).style.height = "320px"
  })

  await expect(page.getByText("Topic metadata")).toBeVisible()
  await expect(page.locator("[data-ab-thread-map-hotspot='true']")).toBeAttached()
  await expect(page.locator("[data-ab-thread-map='true']")).toHaveAttribute("data-visible", "false")

  await page.locator("[data-ab-thread-map-hotspot='true']").hover()
  await expect(page.locator("[data-ab-thread-map='true']")).toHaveAttribute("data-visible", "true")
  await expect(page.locator("[data-ab-thread-map='true']")).toBeVisible()
  await expect(page.locator("[data-ab-thread-map-marker]").first()).toBeVisible()
})
