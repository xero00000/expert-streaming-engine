package com.example.llama

import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.text.format.Formatter
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.getSystemService
import com.example.llama.ui.theme.LlamaAndroidTheme

class MainActivity : ComponentActivity() {
    private val activityManager by lazy { getSystemService<ActivityManager>()!! }
    private val clipboardManager by lazy { getSystemService<ClipboardManager>()!! }
    private val viewModel: MainViewModel by viewModels()

    private fun memoryInfo(): ActivityManager.MemoryInfo =
        ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build(),
        )

        val modelPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                runCatching {
                    contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                viewModel.loadUri(contentResolver, uri)
            }
        }

        val mem = memoryInfo()
        viewModel.log(
            "Memory available: ${Formatter.formatFileSize(this, mem.availMem)} / " +
                Formatter.formatFileSize(this, mem.totalMem),
        )
        viewModel.refreshBackends()

        setContent {
            LlamaAndroidTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    ExpertStreamingScreen(
                        viewModel = viewModel,
                        onPickModel = { modelPicker.launch(arrayOf("application/octet-stream", "*/*")) },
                        onCopy = {
                            clipboardManager.setPrimaryClip(
                                ClipData.newPlainText("Expert Streaming Engine", viewModel.messages.joinToString("\n")),
                            )
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, enabled: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, enabled = enabled, onCheckedChange = onChange)
    }
}

@Composable
private fun IntField(
    label: String,
    value: Int,
    enabled: Boolean,
    allowNegative: Boolean = false,
    onValue: (Int) -> Unit,
) {
    OutlinedTextField(
        modifier = Modifier.weight(1f),
        value = value.toString(),
        enabled = enabled,
        singleLine = true,
        label = { Text(label) },
        onValueChange = { text ->
            if (text.isEmpty() || (allowNegative && text == "-")) return@OutlinedTextField
            text.toIntOrNull()?.let(onValue)
        },
    )
}

@Composable
fun ExpertStreamingScreen(
    viewModel: MainViewModel,
    onPickModel: () -> Unit,
    onCopy: () -> Unit,
) {
    val config = viewModel.config
    val settingsEnabled = !viewModel.isBusy && viewModel.loadedModel == null
    val logState = rememberLazyListState()

    LaunchedEffect(viewModel.messages.size) {
        if (viewModel.messages.isNotEmpty()) logState.animateScrollToItem(viewModel.messages.lastIndex)
    }

    Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
        Text("Expert Streaming Engine", style = MaterialTheme.typography.headlineSmall)
        Text("Galaxy S25 Ultra · GGUF / mmap / MoE / QNN HTP", style = MaterialTheme.typography.bodySmall)
        Text(viewModel.backendStatus, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(vertical = 6.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = onPickModel, enabled = settingsEnabled) { Text("Pick GGUF") }
            Button(onClick = viewModel::unload, enabled = !viewModel.isBusy && viewModel.loadedModel != null) { Text("Unload") }
            Button(onClick = viewModel::refreshBackends, enabled = !viewModel.isBusy) { Text("Backends") }
        }

        Text(
            viewModel.loadedModel?.let { "Loaded: $it" } ?: "No model loaded",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(vertical = 4.dp),
        )

        HorizontalDivider()
        Text("Engine", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 6.dp))

        ToggleRow("Qualcomm QNN / Hexagon HTP", config.useQnn, settingsEnabled) {
            viewModel.updateConfig { c -> c.copy(useQnn = it) }
        }
        ToggleRow("Defer experts (mmap)", config.deferExperts, settingsEnabled) {
            viewModel.updateConfig { c -> c.copy(deferExperts = it) }
        }
        ToggleRow("Route-aware expert prefetch", config.prefetchExperts, settingsEnabled) {
            viewModel.updateConfig { c -> c.copy(prefetchExperts = it) }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField("Context", config.contextSize, settingsEnabled) {
                viewModel.updateConfig { c -> c.copy(contextSize = it.coerceAtLeast(512)) }
            }
            IntField("Threads (0=auto)", config.threads, settingsEnabled) {
                viewModel.updateConfig { c -> c.copy(threads = it.coerceAtLeast(0)) }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField("Batch", config.batchSize, settingsEnabled) {
                viewModel.updateConfig { c -> c.copy(batchSize = it.coerceAtLeast(32)) }
            }
            IntField("Ubatch", config.ubatchSize, settingsEnabled) {
                viewModel.updateConfig { c -> c.copy(ubatchSize = it.coerceAtLeast(1)) }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField("Prefetch threads", config.prefetchThreads, settingsEnabled) {
                viewModel.updateConfig { c -> c.copy(prefetchThreads = it) }
            }
            IntField("Accelerator layers", config.gpuLayers, settingsEnabled) {
                viewModel.updateConfig { c -> c.copy(gpuLayers = it.coerceAtLeast(0)) }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField("CPU MoE (-1=auto)", config.cpuMoeLayers, settingsEnabled, allowNegative = true) {
                viewModel.updateConfig { c -> c.copy(cpuMoeLayers = it.coerceAtLeast(-1)) }
            }
            IntField("Max output", config.maxTokens, settingsEnabled) {
                viewModel.updateConfig { c -> c.copy(maxTokens = it.coerceAtLeast(1)) }
            }
        }

        HorizontalDivider(modifier = Modifier.padding(top = 6.dp))

        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            LazyColumn(state = logState, modifier = Modifier.fillMaxSize()) {
                items(viewModel.messages) { line ->
                    Text(line, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(vertical = 2.dp))
                }
            }
        }

        OutlinedTextField(
            modifier = Modifier.fillMaxWidth(),
            value = viewModel.message,
            onValueChange = viewModel::updateMessage,
            enabled = viewModel.loadedModel != null && !viewModel.isBusy,
            singleLine = false,
            maxLines = 4,
            label = { Text("Prompt") },
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 6.dp)) {
            Button(onClick = viewModel::send, enabled = viewModel.loadedModel != null && !viewModel.isBusy) { Text("Send") }
            Button(onClick = { viewModel.bench() }, enabled = viewModel.loadedModel != null && !viewModel.isBusy) { Text("Bench") }
            Button(onClick = viewModel::clear, enabled = !viewModel.isBusy) { Text("Clear") }
            Button(onClick = onCopy, enabled = viewModel.messages.isNotEmpty()) { Text("Copy") }
        }
    }
}
