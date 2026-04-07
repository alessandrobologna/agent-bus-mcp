import { expect, test } from "@playwright/test"

test("opens topics in tabs and navigates from global search", async ({ page }) => {
  await page.goto("/")

  await expect(page.getByText("Alpha review")).toBeVisible()
  await page.getByRole("button", { name: /Alpha review/i }).click()
  await expect(page.getByText("hello from alpha")).toBeVisible()

  await page.getByRole("button", { name: /^Search$/i }).click()
  await page.getByPlaceholder("Search all messages").fill("handoff")
  const dialog = page.getByRole("dialog", { name: /Search messages across topics/i })
  const result = dialog.getByRole("button", { name: /Beta thread/i })
  await expect(result).toBeVisible()
  await result.click()

  await expect(page).toHaveURL(/\/topics\/.+\?focus=/)
  await expect(page.getByText(/focused context window/i)).toBeVisible()
  await expect(page.getByText("beta handoff summary")).toBeVisible()
})
