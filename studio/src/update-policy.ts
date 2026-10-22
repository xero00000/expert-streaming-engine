export function manualUpdateInstructions(bundleType: string | null): string | null {
  if (bundleType === "deb" || bundleType === "rpm") {
    return "Update ESE Studio with your system’s package manager using the matching CPU or CUDA package from GitHub Releases.";
  }
  if (bundleType === "appimage" || bundleType === "msi" || bundleType === "nsis") return null;
  return "This installation format could not be identified. Install the matching ESE Studio package from GitHub Releases to update.";
}

export async function checkPackageUpdate<T>(
  getBundleType: () => Promise<string | null>,
  check: () => Promise<T | null>,
): Promise<{ kind: "manual"; instructions: string } | { kind: "automatic"; update: T | null }> {
  const instructions = manualUpdateInstructions(await getBundleType());
  if (instructions) return { kind: "manual", instructions };
  return { kind: "automatic", update: await check() };
}
