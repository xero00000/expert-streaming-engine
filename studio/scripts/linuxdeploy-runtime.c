#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Tauri clears bytes 8-10 in this executable's ELF padding. An ELF launcher
// also lets the pinned linuxdeploy receive exclusions absent from Tauri's CLI.
int main(int argc, char ** argv) {
    const int plugin_mode = getenv("LINUXDEPLOY_PLUGIN_MODE") != NULL;
    // A plugin callback already runs inside the extracted AppImage. Reuse
    // its linuxdeploy executable so nested extraction cannot remove the
    // outer invocation's temporary directory.
    const char * real = getenv(plugin_mode ? "LINUXDEPLOY" : "ESE_REAL_LINUXDEPLOY");
    if (!real || !*real) {
        fputs("The real linuxdeploy executable is missing.\n", stderr);
        return 1;
    }
    if (!strcmp(real, argv[0])) {
        fputs("linuxdeploy launcher recursion is not allowed.\n", stderr);
        return 1;
    }
    const char * excluded[] = {
        "libcuda.so*", "libnvidia-*", "libcudart.so*", "libcublas.so*",
        "libcublasLt.so*", "libnccl.so*", "libggml.so*", "libllama.so*", "libmtmd.so*", "libgomp.so*",
    };
    const size_t count = sizeof(excluded) / sizeof(excluded[0]);
    char ** args = calloc((size_t) argc + count * 2 + 2, sizeof(*args));
    if (!args) return 1;
    size_t next = 0;
    args[next++] = (char *) real;
    if (!plugin_mode) args[next++] = "--appimage-extract-and-run";
    for (size_t i = 0; i < count; ++i) {
        args[next++] = "--exclude-library";
        args[next++] = (char *) excluded[i];
    }
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--appimage-extract-and-run")) args[next++] = argv[i];
    }
    args[next] = NULL;
    execv(real, args);
    const int status = errno == ENOENT ? 127 : 1;
    perror("Could not run the pinned linuxdeploy");
    free(args);
    return status;
}
