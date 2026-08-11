plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val expertVulkan = providers.gradleProperty("expertVulkan")
    .orElse("false")
    .map(String::toBoolean)

val expertQnn = providers.gradleProperty("expertQnn")
    .orElse("false")
    .map(String::toBoolean)

// Prefer an explicit Gradle property so Android Studio builds are reproducible,
// but accept Qualcomm's conventional QNN_SDK_ROOT environment variable too.
val qnnSdkRoot = providers.gradleProperty("qnnSdkRoot")
    .orElse(providers.environmentVariable("QNN_SDK_ROOT"))
    .orElse("")

android {
    namespace = "android.llama.cpp"
    compileSdk = 34

    defaultConfig {
        minSdk = 33

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            // The S25 Ultra is arm64 only for our first optimized target. Keeping
            // one ABI also avoids packaging several copies of the large native runtime.
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_VULKAN=${if (expertVulkan.get()) "ON" else "OFF"}",
                    "-DEXPERT_ANDROID_QNN=${if (expertQnn.get()) "ON" else "OFF"}",
                    "-DGGML_OPENMP=OFF",
                    "-DGGML_BUILD_TESTS=OFF",
                    "-DGGML_BUILD_EXAMPLES=OFF",
                )
                if (expertQnn.get()) {
                    require(qnnSdkRoot.get().isNotBlank()) {
                        "-PexpertQnn=true requires -PqnnSdkRoot=/path/to/qairt or QNN_SDK_ROOT"
                    }
                    arguments += "-DQNN_SDK_ROOT=${qnnSdkRoot.get()}"
                }

                // Safe ARMv8.2 baseline with dot-product/FP16 acceleration. A later
                // S25-only flavor can raise this after device-side ISA validation.
                cppFlags += listOf("-O3", "-march=armv8.2-a+dotprod+fp16")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
