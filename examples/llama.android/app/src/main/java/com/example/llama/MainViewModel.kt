package com.example.llama

import android.content.ContentResolver
import android.llama.cpp.LLamaAndroid
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class MainViewModel(
    private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance(),
) : ViewModel() {
    private val tag: String? = this::class.simpleName

    var messages by mutableStateOf(listOf("Expert Streaming Engine Android"))
        private set
    var message by mutableStateOf("")
        private set
    var loadedModel by mutableStateOf<String?>(null)
        private set
    var backendStatus by mutableStateOf("Backend status not queried")
        private set
    var isBusy by mutableStateOf(false)
        private set

    var config by mutableStateOf(LLamaAndroid.EngineConfig())
        private set

    private var modelDocument: ParcelFileDescriptor? = null

    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch {
            runCatching { llamaAndroid.unload() }
            modelDocument?.close()
            modelDocument = null
        }
    }

    fun updateConfig(transform: (LLamaAndroid.EngineConfig) -> LLamaAndroid.EngineConfig) {
        if (!isBusy && loadedModel == null) config = transform(config)
    }

    fun refreshBackends() {
        viewModelScope.launch {
            val qnn = runCatching { llamaAndroid.qnnStatus(config.qnnDspLibraryPath) }
                .getOrElse { "QNN: ${it.message}" }
            val all = runCatching { llamaAndroid.backendSummary() }.getOrElse { "unknown" }
            backendStatus = "Registered: $all\n$qnn"
            messages += backendStatus
        }
    }

    fun loadUri(contentResolver: ContentResolver, uri: Uri) {
        if (isBusy || loadedModel != null) return
        viewModelScope.launch {
            isBusy = true
            var fd: ParcelFileDescriptor? = null
            try {
                fd = contentResolver.openFileDescriptor(uri, "r")
                    ?: throw IllegalStateException("Unable to open selected model")
                val fdPath = "/proc/self/fd/${fd.fd}"
                llamaAndroid.load(fdPath, config)
                modelDocument = fd
                fd = null
                loadedModel = uri.toString()
                messages += "Loaded GGUF through SAF: $uri"
                backendStatus = "Registered: ${llamaAndroid.backendSummary()}\n" +
                    llamaAndroid.qnnStatus(config.qnnDspLibraryPath)
            } catch (t: Throwable) {
                Log.e(tag, "loadUri() failed", t)
                messages += "Load failed: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                runCatching { fd?.close() }
                isBusy = false
            }
        }
    }

    fun load(pathToModel: String) {
        if (isBusy || loadedModel != null) return
        viewModelScope.launch {
            isBusy = true
            try {
                llamaAndroid.load(pathToModel, config)
                loadedModel = pathToModel
                messages += "Loaded $pathToModel"
                backendStatus = "Registered: ${llamaAndroid.backendSummary()}\n" +
                    llamaAndroid.qnnStatus(config.qnnDspLibraryPath)
            } catch (t: Throwable) {
                Log.e(tag, "load() failed", t)
                messages += "Load failed: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                isBusy = false
            }
        }
    }

    fun unload() {
        if (isBusy || loadedModel == null) return
        viewModelScope.launch {
            isBusy = true
            try {
                llamaAndroid.unload()
                modelDocument?.close()
                modelDocument = null
                messages += "Model unloaded"
                loadedModel = null
            } catch (t: Throwable) {
                Log.e(tag, "unload() failed", t)
                messages += "Unload failed: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                isBusy = false
            }
        }
    }

    fun send() {
        val text = message.trim()
        if (text.isEmpty() || isBusy || loadedModel == null) return
        message = ""
        messages += "You: $text"
        messages += ""
        isBusy = true

        viewModelScope.launch {
            llamaAndroid.send(text)
                .catch {
                    Log.e(tag, "send() failed", it)
                    messages = messages.dropLast(1) + "Generation failed: ${it.message}"
                }
                .collect { token ->
                    messages = messages.dropLast(1) + (messages.last() + token)
                }
            isBusy = false
        }
    }

    fun bench(pp: Int = 128, tg: Int = 32, pl: Int = 1, nr: Int = 1) {
        if (isBusy || loadedModel == null) return
        viewModelScope.launch {
            isBusy = true
            try {
                messages += llamaAndroid.bench(pp, tg, pl, nr)
            } catch (t: Throwable) {
                Log.e(tag, "bench() failed", t)
                messages += "Benchmark failed: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                isBusy = false
            }
        }
    }

    fun updateMessage(newMessage: String) { message = newMessage }
    fun clear() { messages = emptyList() }
    fun log(value: String) { messages += value }
}
