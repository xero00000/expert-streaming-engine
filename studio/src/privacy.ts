export function privacySafeError(value: string) {
  const scrubUrl = (segment: string) => {
    try {
      const url = new URL(segment);
      if (url.protocol === "file:") return "[local file URL]";
      const query = url.search ? "?[redacted]" : "";
      const fragment = url.hash ? "#[redacted]" : "";
      // Rebuild from protocol and host so URL userinfo is never retained.
      return `${url.protocol}//${url.host}${url.pathname}${query}${fragment}`;
    } catch {
      return "[redacted URL]";
    }
  };

  const scrubLocalDetails = (segment: string) => segment
    .replace(/(["'])(?:[A-Za-z]:[\\/]|\\\\)[^\r\n"'<>|]*\1/g, "$1[local Windows path]$1")
    .replace(/(^|[\s("'=:\[{,;])(?:[A-Za-z]:[\\/]|\\\\)[^\r\n;,"'<>|)\]}]+/gm, "$1[local Windows path]")
    // Match absolute Unix paths only at a textual boundary. This covers mounted
    // model paths such as /run/media/... and /mnt/... as well as home directories.
    .replace(/(["'])\/(?!\/)[^\r\n"'<>|]*\1/g, "$1[local path]$1")
    .replace(/(^|[\s("'=:\[{,;])\/(?!\/)[^\r\n;,"'<>|)\]}]+/gm, "$1[local path]")
    .replace(/(["']?(?:username|user|hostname|host|account|email|computer_name|machine_name)["']?\s*:\s*)(?:"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[^,\s}\]\r\n]+)/gi, "$1\"[redacted]\"")
    .replace(/\b(username|user|hostname|host|account|email|computer_name|machine_name)\b\s*=\s*[^\s,;]+/gi, "$1=[redacted]");

  // Keep URL origin/path details actionable while removing credentials, signed
  // queries, and fragments. Splitting first also prevents drive-like URL path
  // segments (for example /C:/docs) from being treated as local files.
  return value
    .split(/((?:https?|file):\/\/[^\s"'<>]+)/gi)
    .map((segment, index) => index % 2 === 1 ? scrubUrl(segment) : scrubLocalDetails(segment))
    .join("")
    .slice(0, 800);
}
