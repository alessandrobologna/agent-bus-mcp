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

test("renders the thread map for long desktop topics and toggles to the inspector", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto("/")

  await page.getByRole("button", { name: /Alpha review/i }).click()

  await expect(page.getByRole("tab", { name: "Map" })).toBeVisible()
  await expect(page.locator("[data-ab-thread-map='true']")).toBeVisible()
  await expect(page.locator("[data-ab-thread-map-marker]").first()).toBeVisible()

  await page.getByRole("tab", { name: "Inspector" }).click()
  await expect(page.getByText("Topic metadata")).toBeVisible()

  await page.getByRole("tab", { name: "Map" }).click()
  await expect(page.locator("[data-ab-thread-map='true']")).toBeVisible()
})
