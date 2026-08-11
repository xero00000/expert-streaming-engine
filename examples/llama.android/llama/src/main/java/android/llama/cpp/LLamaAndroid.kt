package android.llama.cpp

import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class LLamaAndroid {
    data class EngineConfig(
        val contextSize: Int = 4096,
        val threads: Int = 0,
        val batchSize: Int = 512,
        val ubatchSize: Int = 256,
        val maxTokens: Int = 128,
        val deferExperts: Boolean = true,
        val prefetchExperts: Boolean = true,
        val prefetchThreads: Int = 2,
        val gpuLayers: Int = 0,
        val cpuMoeLayers: Int = -1,
        val useQnn: Boolean = true,
        val qnnDspLibraryPath: String? = null,
    )

    private val tag: String? = this::class.simpleName
    private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.Idle }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "ExpertStreaming-RunLoop") {
            Log.d(tag, "Dedicated native run loop: ${Thread.currentThread().name}")
            System.loadLibrary("llama-android")
            log_to_android()
            backend_init()
            Log.d(tag, system_info())
            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled native run-loop exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    private external fun log_to_android()
    private external fun load_engine_model(
        filename: String,
        deferExperts: Boolean,
        nGpuLayers: Int,
        nCpuMoe: Int,
        useQnn: Boolean,
    ): Long
    private external fun free_model(model: Long)
    private external fun new_engine_context(
        model: Long,
        nCtx: Int,
        nThreads: Int,
        nBatch: Int,
        nUbatch: Int,
        prefetchExperts: Boolean,
        prefetchThreads: Int,
        useQnn: Boolean,
    ): Long
    private external fun free_context(context: Long)
    private external fun backend_init()
    private external fun free_batch(batch: Long)
    private external fun new_batch(nTokens: Int, embd: Int, nSeqMax: Int): Long
    private external fun bench_model(
        context: Long,
        model: Long,
        batch: Long,
        pp: Int,
        tg: Int,
        pl: Int,
        nr: Int,
    ): String
    private external fun system_info(): String
    private external fun qnn_probe(): String
    private external fun backend_summary(): String
    private external fun set_dsp_library_path(path: String)
    private external fun completion_init(
        context: Long,
        batch: Long,
        text: String,
        nLen: Int,
    ): Int
    private external fun completion_loop(
        context: Long,
        batch: Long,
        nLen: Int,
        ncur: IntVar,
    ): String?
    private external fun kv_cache_clear(context: Long)

    suspend fun systemInfo(): String = withContext(runLoop) { system_info() }

    suspend fun qnnStatus(dspLibraryPath: String? = null): String = withContext(runLoop) {
        dspLibraryPath?.takeIf { it.isNotBlank() }?.let { set_dsp_library_path(it) }
        qnn_probe()
    }

    suspend fun backendSummary(): String = withContext(runLoop) { backend_summary() }

    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
        return withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> bench_model(state.context, state.model, state.batch, pp, tg, pl, nr)
                else -> throw IllegalStateException("No model loaded")
            }
        }
    }

    suspend fun load(pathToModel: String, config: EngineConfig = EngineConfig()) {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Idle -> {
                    config.qnnDspLibraryPath
                        ?.takeIf { it.isNotBlank() }
                        ?.let { set_dsp_library_path(it) }

                    val model = load_engine_model(
                        pathToModel,
                        config.deferExperts,
                        config.gpuLayers,
                        config.cpuMoeLayers,
                        config.useQnn,
                    )
                    if (model == 0L) throw IllegalStateException("load_engine_model() failed")

                    val context = try {
                        new_engine_context(
                            model,
                            config.contextSize,
                            config.threads,
                            config.batchSize,
                            config.ubatchSize,
                            config.prefetchExperts,
                            config.prefetchThreads,
                            config.useQnn,
                        )
                    } catch (t: Throwable) {
                        free_model(model)
                        throw t
                    }

                    if (context == 0L) {
                        free_model(model)
                        throw IllegalStateException("new_engine_context() failed")
                    }

                    val batch = new_batch(config.batchSize, 0, 1)
                    if (batch == 0L) {
                        free_context(context)
                        free_model(model)
                        throw IllegalStateException("new_batch() failed")
                    }

                    Log.i(tag, "Loaded $pathToModel with $config; backends=${backend_summary()}")
                    threadLocalState.set(State.Loaded(model, context, batch, config))
                }
                else -> throw IllegalStateException("Model already loaded")
            }
        }
    }

    fun send(message: String): Flow<String> = flow {
        when (val state = threadLocalState.get()) {
            is State.Loaded -> {
                val nLen = state.config.maxTokens
                val ncur = IntVar(completion_init(state.context, state.batch, message, nLen))
                while (ncur.value <= nLen) {
                    val str = completion_loop(state.context, state.batch, nLen, ncur) ?: break
                    emit(str)
                }
                kv_cache_clear(state.context)
            }
            else -> throw IllegalStateException("No model loaded")
        }
    }.flowOn(runLoop)

    suspend fun unload() {
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    free_context(state.context)
                    free_model(state.model)
                    free_batch(state.batch)
                    threadLocalState.set(State.Idle)
                }
                else -> Unit
            }
        }
    }

    /** Deterministic teardown for ViewModel.onCleared(). */
    fun unloadBlocking() {
        runBlocking { unload() }
    }

    companion object {
        private class IntVar(value: Int) {
            @Volatile
            var value: Int = value
                private set
            fun inc() { synchronized(this) { value += 1 } }
        }

        private sealed interface State {
            data object Idle : State
            data class Loaded(
                val model: Long,
                val context: Long,
                val batch: Long,
                val config: EngineConfig,
            ) : State
        }

        private val instance = LLamaAndroid()
        fun instance(): LLamaAndroid = instance
    }
}
