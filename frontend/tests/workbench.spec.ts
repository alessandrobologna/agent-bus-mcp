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
  await expect(page.getByText("beta handoff summary")).toBeVisible()
})
