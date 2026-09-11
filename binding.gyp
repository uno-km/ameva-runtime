{
  "variables": {
    "android_ndk_path%": ""
  },
  "targets": [
    {
      "target_name": "ameva_native",
      "sources": [
        "src/bridge/node_api_bridge.c",
        "src/c_api/ameva_vulkan_c_api.cpp",
        "src/doctor/probe_stages.cpp",
        "src/core/vulkan_loader.cpp",
        "src/quirks/adreno_quirks.cpp",
        "src/quirks/mali_quirks.cpp"
      ],
      "include_dirs": [
        "src"
      ],
      "cflags": [
        "-Wall",
        "-Wextra",
        "-O3",
        "-fPIC"
      ],
      "cflags_cc": [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-O3",
        "-fPIC",
        "-fexceptions"
      ],
      "conditions": [
        ["OS=='android'", {
          "libraries": [
            "-lvulkan",
            "-ldl",
            "-llog"
          ]
        }],
        ["OS=='linux'", {
          "libraries": [
            "-lvulkan",
            "-ldl",
            "-lpthread"
          ]
        }]
      ]
    }
  ]
}
