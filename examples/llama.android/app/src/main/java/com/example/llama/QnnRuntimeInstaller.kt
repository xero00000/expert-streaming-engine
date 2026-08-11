package com.example.llama

import android.content.Context
import java.io.File

/**
 * Copies optional QAIRT Hexagon-side runtime files from assets/qnn into
 * app-private storage. QNN's HTP backend discovers these through
 * ADSP_LIBRARY_PATH, which the native bridge sets before the first QNN probe.
 *
 * Host-side Android .so files such as libQnnHtp.so belong in
 * llama/src/main/jniLibs/arm64-v8a so Android's linker can load them normally.
 */
object QnnRuntimeInstaller {
    fun installIfBundled(context: Context): String? {
        val rootEntries = context.assets.list("qnn") ?: return null
        if (rootEntries.isEmpty()) return null

        val destination = File(context.filesDir, "qnn-runtime")
        copyAssetTree(context, "qnn", destination)
        return destination.absolutePath
    }

    private fun copyAssetTree(context: Context, assetPath: String, output: File) {
        val entries = context.assets.list(assetPath) ?: emptyArray()
        if (entries.isEmpty()) {
            output.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                output.outputStream().use { target -> input.copyTo(target) }
            }
            return
        }

        output.mkdirs()
        for (entry in entries) {
            val childAsset = "$assetPath/$entry"
            val childOutput = File(output, entry)
            copyAssetTree(context, childAsset, childOutput)
        }
    }
}
