export function privacySafeError(value: string) {
  const scrubLocalDetails = (segment: string) => segment
    .replace(/(^|[\s("'=:\[{,;])(?:[A-Za-z]:[\\/]|\\\\)[^\r\n"'<>|]*/g, "$1[local Windows path]")
    // Match absolute Unix paths only at a textual boundary. This covers mounted
    // model paths such as /run/media/... and /mnt/... as well as home directories.
    .replace(/(^|[\s("'=:\[{,;])\/(?!\/)[^\s:;,"'<>|)\]}]+/g, "$1[local path]")
    .replace(/\b(username|user|hostname|host)\s*[=:]\s*[^\s,;]+/gi, "$1=[redacted]");

  // Keep web links actionable in the prefilled report. Splitting first also
  // prevents drive-like URL segments (for example /C:/docs) from being treated
  // as local filesystem details.
  return value
    .split(/(https?:\/\/[^\s"'<>]+)/gi)
    .map((segment, index) => index % 2 === 1 ? segment : scrubLocalDetails(segment))
    .join("")
    .slice(0, 800);
}
