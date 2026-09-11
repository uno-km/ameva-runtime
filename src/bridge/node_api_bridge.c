#include <node_api.h>
#include "../c_api/ameva_vulkan_c_api.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>

#define AMEVA_NAPI_BRIDGE_VERSION 1

// Native calls serialization mutex across async worker threads
static pthread_mutex_t g_native_mutex = PTHREAD_MUTEX_INITIALIZER;

// Concurrency instrumentation metrics (strictly tracked INSIDE lock)
static atomic_int g_active_calls = 0;
static atomic_int g_max_active_calls = 0;
static atomic_int g_total_calls = 0;

static void track_call_start(void) {
    int active = atomic_fetch_add(&g_active_calls, 1) + 1;
    int cur_max = atomic_load(&g_max_active_calls);
    while (active > cur_max) {
        if (atomic_compare_exchange_weak(&g_max_active_calls, &cur_max, active)) {
            break;
        }
    }
    atomic_fetch_add(&g_total_calls, 1);
}

static void track_call_end(void) {
    atomic_fetch_sub(&g_active_calls, 1);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool validate_dimension(napi_env env, napi_value val, const char* name, uint32_t* out) {
    (void)name;
    napi_valuetype vt;
    if (napi_typeof(env, val, &vt) != napi_ok || vt != napi_number) {
        napi_throw_type_error(env, "INVALID_DIMENSION_TYPE", "Dimension argument must be a number");
        return false;
    }

    double dval;
    if (napi_get_value_double(env, val, &dval) != napi_ok) {
        napi_throw_type_error(env, "INVALID_DIMENSION", "Failed to parse dimension value");
        return false;
    }

    if (isnan(dval) || isinf(dval)) {
        napi_throw_range_error(env, "DIMENSION_NOT_FINITE", "Dimension cannot be NaN or Infinity");
        return false;
    }

    if (dval <= 0.0) {
        napi_throw_range_error(env, "DIMENSION_NON_POSITIVE", "Dimension must be strictly greater than 0");
        return false;
    }

    if (floor(dval) != dval) {
        napi_throw_range_error(env, "DIMENSION_NOT_INTEGER", "Dimension must be an integer");
        return false;
    }

    if (dval > (double)UINT32_MAX) {
        napi_throw_range_error(env, "DIMENSION_OVERFLOW", "Dimension exceeds UINT32_MAX");
        return false;
    }

    *out = (uint32_t)dval;
    return true;
}

// ---------------------------------------------------------------------------
// 1. Version and Metadata APIs (Synchronous)
// ---------------------------------------------------------------------------

static napi_value Method_GetBridgeVersion(napi_env env, napi_callback_info info) {
    (void)info;
    napi_value result;
    napi_create_uint32(env, AMEVA_NAPI_BRIDGE_VERSION, &result);
    return result;
}

static napi_value Method_GetNativeAbiVersion(napi_env env, napi_callback_info info) {
    (void)info;
    napi_value obj;
    napi_create_object(env, &obj);

    napi_value supported, status, compile_check;
    napi_get_boolean(env, false, &supported);
    napi_create_string_utf8(env, "Runtime native ABI compatibility verification unavailable", NAPI_AUTO_LENGTH, &status);
    napi_create_string_utf8(env, "Validated by compile-time header inclusion", NAPI_AUTO_LENGTH, &compile_check);

    napi_set_named_property(env, obj, "supported", supported);
    napi_set_named_property(env, obj, "status", status);
    napi_set_named_property(env, obj, "compileHeaderCompatibility", compile_check);

    return obj;
}

static napi_value Method_GetDiagnosticResultStructSize(napi_env env, napi_callback_info info) {
    (void)info;
    napi_value result;
    napi_create_uint32(env, (uint32_t)sizeof(AmevaDiagnosticResult), &result);
    return result;
}

static napi_value Method_GetVersion(napi_env env, napi_callback_info info) {
    (void)info;
    const char* ver = ameva_get_version();
    napi_value result;
    napi_create_string_utf8(env, ver ? ver : "unknown", NAPI_AUTO_LENGTH, &result);
    return result;
}

static napi_value Method_IsTensorAligned(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 2) {
        napi_throw_type_error(env, "INVALID_ARG_COUNT", "isTensorAligned requires 2 arguments (ne01, ne11)");
        return NULL;
    }

    uint32_t ne01, ne11;
    if (!validate_dimension(env, argv[0], "ne01", &ne01)) return NULL;
    if (!validate_dimension(env, argv[1], "ne11", &ne11)) return NULL;

    bool aligned = ameva_is_tensor_aligned(ne01, ne11);
    napi_value result;
    napi_get_boolean(env, aligned, &result);
    return result;
}

static napi_value Method_IsVulkanAvailable(napi_env env, napi_callback_info info) {
    (void)info;
    pthread_mutex_lock(&g_native_mutex);
    track_call_start();
    bool avail = ameva_is_vulkan_available();
    track_call_end();
    pthread_mutex_unlock(&g_native_mutex);

    napi_value result;
    napi_get_boolean(env, avail, &result);
    return result;
}

static napi_value Method_QuickProbe(napi_env env, napi_callback_info info) {
    (void)info;
    char device_name[256] = {0};

    pthread_mutex_lock(&g_native_mutex);
    track_call_start();
    bool ok = ameva_quick_probe(device_name, (int)sizeof(device_name));
    track_call_end();
    pthread_mutex_unlock(&g_native_mutex);

    napi_value obj;
    napi_create_object(env, &obj);

    napi_value js_avail, js_name;
    napi_get_boolean(env, ok, &js_avail);
    napi_create_string_utf8(env, device_name, NAPI_AUTO_LENGTH, &js_name);

    napi_set_named_property(env, obj, "available", js_avail);
    napi_set_named_property(env, obj, "deviceName", js_name);

    return obj;
}

// ---------------------------------------------------------------------------
// 2. Concurrency Diagnostics APIs
// ---------------------------------------------------------------------------

static napi_value Method_GetConcurrencyMetrics(napi_env env, napi_callback_info info) {
    (void)info;
    napi_value obj;
    napi_create_object(env, &obj);

    napi_value v_active, v_max, v_total;
    napi_create_int32(env, atomic_load(&g_active_calls), &v_active);
    napi_create_int32(env, atomic_load(&g_max_active_calls), &v_max);
    napi_create_int32(env, atomic_load(&g_total_calls), &v_total);

    napi_set_named_property(env, obj, "activeCalls", v_active);
    napi_set_named_property(env, obj, "maxActiveCalls", v_max);
    napi_set_named_property(env, obj, "totalCalls", v_total);

    return obj;
}

static napi_value Method_ResetConcurrencyMetrics(napi_env env, napi_callback_info info) {
    (void)info;
    atomic_store(&g_active_calls, 0);
    atomic_store(&g_max_active_calls, 0);
    atomic_store(&g_total_calls, 0);

    napi_value ok;
    napi_get_boolean(env, true, &ok);
    return ok;
}

// ---------------------------------------------------------------------------
// 3. Asynchronous runDiagnostic
// ---------------------------------------------------------------------------

typedef struct {
    bool verbose;
    int exit_code;
    AmevaDiagnosticResult result;
    napi_async_work async_work;
    napi_deferred deferred;
} DiagnosticWorkCarrier;

static void Execute_RunDiagnostic(napi_env env, void* data) {
    (void)env;
    DiagnosticWorkCarrier* carrier = (DiagnosticWorkCarrier*)data;

    pthread_mutex_lock(&g_native_mutex);
    track_call_start();
    memset(&carrier->result, 0, sizeof(AmevaDiagnosticResult));
    carrier->exit_code = ameva_run_diagnostic(carrier->verbose, &carrier->result);
    track_call_end();
    pthread_mutex_unlock(&g_native_mutex);
}

static void Complete_RunDiagnostic(napi_env env, napi_status status, void* data) {
    DiagnosticWorkCarrier* carrier = (DiagnosticWorkCarrier*)data;

    if (status != napi_ok) {
        napi_value err_msg, err;
        napi_create_string_utf8(env, "Async diagnostic worker execution failed", NAPI_AUTO_LENGTH, &err_msg);
        napi_create_error(env, NULL, err_msg, &err);
        napi_reject_deferred(env, carrier->deferred, err);
    } else if (carrier->exit_code != 0) {
        // Strict contract: C ABI failure -> Promise reject with native error code. No partial results.
        napi_value err_msg, err_code, code_str, err;
        char msg_buf[128];
        snprintf(msg_buf, sizeof(msg_buf), "Hardware diagnostic failed with native exit code %d", carrier->exit_code);
        napi_create_string_utf8(env, msg_buf, NAPI_AUTO_LENGTH, &err_msg);
        napi_create_string_utf8(env, "HARDWARE_DIAGNOSTIC_FAILED", NAPI_AUTO_LENGTH, &code_str);
        napi_create_int32(env, carrier->exit_code, &err_code);
        napi_create_error(env, code_str, err_msg, &err);
        napi_set_named_property(env, err, "exitCode", err_code);
        napi_reject_deferred(env, carrier->deferred, err);
    } else {
        napi_value obj;
        napi_create_object(env, &obj);

        napi_value val;
        napi_create_string_utf8(env, "compute", NAPI_AUTO_LENGTH, &val);
        napi_set_named_property(env, obj, "diagnosticScope", val);

        napi_get_boolean(env, carrier->result.overall_success, &val);
        napi_set_named_property(env, obj, "scopeSuccess", val);

        napi_get_boolean(env, carrier->result.overall_success, &val);
        napi_set_named_property(env, obj, "overallSuccess", val);

        napi_create_int32(env, carrier->result.passed_stages, &val);
        napi_set_named_property(env, obj, "passedStages", val);

        napi_create_int32(env, carrier->result.total_stages, &val);
        napi_set_named_property(env, obj, "totalStages", val);

        napi_create_double(env, carrier->result.total_elapsed_ms, &val);
        napi_set_named_property(env, obj, "totalElapsedMs", val);

        carrier->result.device_name[sizeof(carrier->result.device_name) - 1] = '\0';
        napi_create_string_utf8(env, carrier->result.device_name, NAPI_AUTO_LENGTH, &val);
        napi_set_named_property(env, obj, "deviceName", val);

        carrier->result.driver_version[sizeof(carrier->result.driver_version) - 1] = '\0';
        napi_create_string_utf8(env, carrier->result.driver_version, NAPI_AUTO_LENGTH, &val);
        napi_set_named_property(env, obj, "driverVersion", val);

        carrier->result.loader_path[sizeof(carrier->result.loader_path) - 1] = '\0';
        napi_create_string_utf8(env, carrier->result.loader_path, NAPI_AUTO_LENGTH, &val);
        napi_set_named_property(env, obj, "loaderPath", val);

        carrier->result.recommended_backend[sizeof(carrier->result.recommended_backend) - 1] = '\0';
        napi_create_string_utf8(env, carrier->result.recommended_backend, NAPI_AUTO_LENGTH, &val);
        napi_set_named_property(env, obj, "recommendedBackend", val);

        napi_get_boolean(env, carrier->result.compute_certified, &val);
        napi_set_named_property(env, obj, "computeCertified", val);

        napi_get_boolean(env, carrier->result.model_certified, &val);
        napi_set_named_property(env, obj, "modelCertified", val);

        napi_create_int32(env, carrier->exit_code, &val);
        napi_set_named_property(env, obj, "exitCode", val);

        napi_resolve_deferred(env, carrier->deferred, obj);
    }

    napi_delete_async_work(env, carrier->async_work);
    free(carrier);
}

static napi_value Method_RunDiagnostic(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    bool verbose = false;

    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) == napi_ok && argc >= 1) {
        napi_valuetype vt;
        if (napi_typeof(env, argv[0], &vt) == napi_ok) {
            if (vt == napi_boolean) {
                napi_get_value_bool(env, argv[0], &verbose);
            } else if (vt == napi_object) {
                napi_value v_prop;
                if (napi_get_named_property(env, argv[0], "verbose", &v_prop) == napi_ok) {
                    napi_get_value_bool(env, v_prop, &verbose);
                }
            }
        }
    }

    DiagnosticWorkCarrier* carrier = (DiagnosticWorkCarrier*)calloc(1, sizeof(DiagnosticWorkCarrier));
    if (!carrier) {
        napi_throw_error(env, "OUT_OF_MEMORY", "Failed to allocate memory for diagnostic carrier");
        return NULL;
    }
    carrier->verbose = verbose;

    napi_value promise;
    if (napi_create_promise(env, &carrier->deferred, &promise) != napi_ok) {
        free(carrier);
        napi_throw_error(env, "PROMISE_CREATION_FAILED", "Failed to create promise");
        return NULL;
    }

    napi_value work_name;
    napi_create_string_utf8(env, "ameva_run_diagnostic_work", NAPI_AUTO_LENGTH, &work_name);

    if (napi_create_async_work(env, NULL, work_name,
                               Execute_RunDiagnostic,
                               Complete_RunDiagnostic,
                               carrier,
                               &carrier->async_work) != napi_ok) {
        free(carrier);
        napi_throw_error(env, "ASYNC_WORK_CREATION_FAILED", "Failed to create async work");
        return NULL;
    }

    if (napi_queue_async_work(env, carrier->async_work) != napi_ok) {
        napi_delete_async_work(env, carrier->async_work);
        free(carrier);
        napi_throw_error(env, "ASYNC_WORK_QUEUE_FAILED", "Failed to queue async work");
        return NULL;
    }

    return promise;
}

// ---------------------------------------------------------------------------
// 4. Asynchronous matmulF32 (Strict CPU Reference Implementation)
// ---------------------------------------------------------------------------

typedef struct {
    float* a;
    float* b;
    float* c;
    int m;
    int k;
    int n;
    size_t mn_elements;
    int exit_code;
    bool with_telemetry;
    napi_async_work async_work;
    napi_deferred deferred;
} MatmulWorkCarrier;

static void Execute_MatmulF32(napi_env env, void* data) {
    (void)env;
    MatmulWorkCarrier* carrier = (MatmulWorkCarrier*)data;

    pthread_mutex_lock(&g_native_mutex);
    track_call_start();
    carrier->exit_code = ameva_matmul_f32(carrier->a, carrier->b, carrier->c,
                                          carrier->m, carrier->k, carrier->n);
    track_call_end();
    pthread_mutex_unlock(&g_native_mutex);
}

static void Complete_MatmulF32(napi_env env, napi_status status, void* data) {
    MatmulWorkCarrier* carrier = (MatmulWorkCarrier*)data;

    if (status != napi_ok) {
        napi_value err_msg, err;
        napi_create_string_utf8(env, "Async matmul worker execution failed", NAPI_AUTO_LENGTH, &err_msg);
        napi_create_error(env, NULL, err_msg, &err);
        napi_reject_deferred(env, carrier->deferred, err);
    } else if (carrier->exit_code != 0) {
        napi_value err_msg, err_code, code_str, err;
        char msg_buf[128];
        snprintf(msg_buf, sizeof(msg_buf), "Native matmul failed with exit code %d", carrier->exit_code);
        napi_create_string_utf8(env, msg_buf, NAPI_AUTO_LENGTH, &err_msg);
        napi_create_string_utf8(env, "NATIVE_MATMUL_FAILED", NAPI_AUTO_LENGTH, &code_str);
        napi_create_int32(env, carrier->exit_code, &err_code);
        napi_create_error(env, code_str, err_msg, &err);
        napi_set_named_property(env, err, "exitCode", err_code);
        napi_reject_deferred(env, carrier->deferred, err);
    } else {
        size_t byte_length = carrier->mn_elements * sizeof(float);
        void* ab_data = NULL;
        napi_value ab;
        if (napi_create_arraybuffer(env, byte_length, &ab_data, &ab) == napi_ok && ab_data) {
            memcpy(ab_data, carrier->c, byte_length);

            napi_value result_typedarray;
            if (napi_create_typedarray(env, napi_float32_array, carrier->mn_elements, ab, 0, &result_typedarray) == napi_ok) {
                if (carrier->with_telemetry) {
                    napi_value res_obj;
                    napi_create_object(env, &res_obj);
                    napi_set_named_property(env, res_obj, "data", result_typedarray);

                    napi_value v_m, v_k, v_n;
                    napi_create_int32(env, carrier->m, &v_m);
                    napi_create_int32(env, carrier->k, &v_k);
                    napi_create_int32(env, carrier->n, &v_n);
                    napi_set_named_property(env, res_obj, "m", v_m);
                    napi_set_named_property(env, res_obj, "k", v_k);
                    napi_set_named_property(env, res_obj, "n", v_n);

                    napi_value tel_obj;
                    napi_create_object(env, &tel_obj);
                    napi_value v_backend, v_impl, v_accel, v_dispatch, v_duration;
                    napi_create_string_utf8(env, "cpu_reference", NAPI_AUTO_LENGTH, &v_backend);
                    napi_create_string_utf8(env, "native_c_reference_gemm", NAPI_AUTO_LENGTH, &v_impl);
                    napi_get_boolean(env, false, &v_accel);
                    napi_create_string_utf8(env, "NOT_PERFORMED", NAPI_AUTO_LENGTH, &v_dispatch);
                    napi_create_double(env, 0.0, &v_duration);

                    napi_set_named_property(env, tel_obj, "backend", v_backend);
                    napi_set_named_property(env, tel_obj, "implementation", v_impl);
                    napi_set_named_property(env, tel_obj, "accelerated", v_accel);
                    napi_set_named_property(env, tel_obj, "vulkanDispatch", v_dispatch);
                    napi_set_named_property(env, tel_obj, "durationMs", v_duration);

                    napi_set_named_property(env, res_obj, "telemetry", tel_obj);

                    napi_resolve_deferred(env, carrier->deferred, res_obj);
                } else {
                    napi_resolve_deferred(env, carrier->deferred, result_typedarray);
                }
            } else {
                napi_value err_msg, err;
                napi_create_string_utf8(env, "Failed to create output Float32Array", NAPI_AUTO_LENGTH, &err_msg);
                napi_create_error(env, NULL, err_msg, &err);
                napi_reject_deferred(env, carrier->deferred, err);
            }
        } else {
            napi_value err_msg, err;
            napi_create_string_utf8(env, "Failed to allocate output ArrayBuffer", NAPI_AUTO_LENGTH, &err_msg);
            napi_create_error(env, NULL, err_msg, &err);
            napi_reject_deferred(env, carrier->deferred, err);
        }
    }

    napi_delete_async_work(env, carrier->async_work);
    free(carrier->a);
    free(carrier->b);
    free(carrier->c);
    free(carrier);
}

static napi_value Common_MatmulF32(napi_env env, napi_callback_info info, bool with_telemetry) {
    size_t argc = 5;
    napi_value argv[5];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 5) {
        napi_throw_type_error(env, "INVALID_ARG_COUNT", "matmulF32 requires 5 arguments (a, b, m, k, n)");
        return NULL;
    }

    uint32_t m, k, n;
    if (!validate_dimension(env, argv[2], "m", &m)) return NULL;
    if (!validate_dimension(env, argv[3], "k", &k)) return NULL;
    if (!validate_dimension(env, argv[4], "n", &n)) return NULL;

    uint64_t mk = (uint64_t)m * (uint64_t)k;
    if (k > 0 && mk / k != m) {
        napi_throw_range_error(env, "OVERFLOW", "m * k dimension overflow");
        return NULL;
    }

    uint64_t kn = (uint64_t)k * (uint64_t)n;
    if (n > 0 && kn / n != k) {
        napi_throw_range_error(env, "OVERFLOW", "k * n dimension overflow");
        return NULL;
    }

    uint64_t mn = (uint64_t)m * (uint64_t)n;
    if (n > 0 && mn / n != m) {
        napi_throw_range_error(env, "OVERFLOW", "m * n dimension overflow");
        return NULL;
    }

    if (mk > SIZE_MAX / sizeof(float) || kn > SIZE_MAX / sizeof(float) || mn > SIZE_MAX / sizeof(float)) {
        napi_throw_range_error(env, "OVERFLOW", "Byte size calculation overflow");
        return NULL;
    }

    bool is_typedarray = false;
    if (napi_is_typedarray(env, argv[0], &is_typedarray) != napi_ok || !is_typedarray) {
        napi_throw_type_error(env, "INVALID_TYPE", "Argument 'a' must be a Float32Array");
        return NULL;
    }
    napi_typedarray_type type_a;
    size_t len_a = 0;
    void* data_a = NULL;
    if (napi_get_typedarray_info(env, argv[0], &type_a, &len_a, &data_a, NULL, NULL) != napi_ok || type_a != napi_float32_array) {
        napi_throw_type_error(env, "INVALID_TYPE", "Argument 'a' must be Float32Array");
        return NULL;
    }
    if (len_a < mk) {
        napi_throw_range_error(env, "BUFFER_TOO_SMALL", "Argument 'a' Float32Array length is smaller than m * k");
        return NULL;
    }

    if (napi_is_typedarray(env, argv[1], &is_typedarray) != napi_ok || !is_typedarray) {
        napi_throw_type_error(env, "INVALID_TYPE", "Argument 'b' must be a Float32Array");
        return NULL;
    }
    napi_typedarray_type type_b;
    size_t len_b = 0;
    void* data_b = NULL;
    if (napi_get_typedarray_info(env, argv[1], &type_b, &len_b, &data_b, NULL, NULL) != napi_ok || type_b != napi_float32_array) {
        napi_throw_type_error(env, "INVALID_TYPE", "Argument 'b' must be Float32Array");
        return NULL;
    }
    if (len_b < kn) {
        napi_throw_range_error(env, "BUFFER_TOO_SMALL", "Argument 'b' Float32Array length is smaller than k * n");
        return NULL;
    }

    float* a_copy = (float*)malloc(mk * sizeof(float));
    float* b_copy = (float*)malloc(kn * sizeof(float));
    float* c_buf  = (float*)calloc(mn, sizeof(float));
    MatmulWorkCarrier* carrier = (MatmulWorkCarrier*)calloc(1, sizeof(MatmulWorkCarrier));

    if (!a_copy || !b_copy || !c_buf || !carrier) {
        free(a_copy);
        free(b_copy);
        free(c_buf);
        free(carrier);
        napi_throw_error(env, "OUT_OF_MEMORY", "Failed to allocate memory for matmul execution");
        return NULL;
    }

    memcpy(a_copy, data_a, mk * sizeof(float));
    memcpy(b_copy, data_b, kn * sizeof(float));

    carrier->a = a_copy;
    carrier->b = b_copy;
    carrier->c = c_buf;
    carrier->m = (int)m;
    carrier->k = (int)k;
    carrier->n = (int)n;
    carrier->mn_elements = (size_t)mn;
    carrier->with_telemetry = with_telemetry;

    napi_value promise;
    if (napi_create_promise(env, &carrier->deferred, &promise) != napi_ok) {
        free(a_copy);
        free(b_copy);
        free(c_buf);
        free(carrier);
        napi_throw_error(env, "PROMISE_CREATION_FAILED", "Failed to create promise for matmul");
        return NULL;
    }

    napi_value work_name;
    napi_create_string_utf8(env, "ameva_matmul_f32_work", NAPI_AUTO_LENGTH, &work_name);

    if (napi_create_async_work(env, NULL, work_name,
                               Execute_MatmulF32,
                               Complete_MatmulF32,
                               carrier,
                               &carrier->async_work) != napi_ok) {
        free(a_copy);
        free(b_copy);
        free(c_buf);
        free(carrier);
        napi_throw_error(env, "ASYNC_WORK_CREATION_FAILED", "Failed to create async work for matmul");
        return NULL;
    }

    if (napi_queue_async_work(env, carrier->async_work) != napi_ok) {
        napi_delete_async_work(env, carrier->async_work);
        free(a_copy);
        free(b_copy);
        free(c_buf);
        free(carrier);
        napi_throw_error(env, "ASYNC_WORK_QUEUE_FAILED", "Failed to queue async work for matmul");
        return NULL;
    }

    return promise;
}

static napi_value Method_MatmulF32(napi_env env, napi_callback_info info) {
    return Common_MatmulF32(env, info, false);
}

static napi_value Method_MatmulF32WithTelemetry(napi_env env, napi_callback_info info) {
    return Common_MatmulF32(env, info, true);
}

// ---------------------------------------------------------------------------
// Module Registration
// ---------------------------------------------------------------------------

static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        { "getBridgeVersion", 0, Method_GetBridgeVersion, 0, 0, 0, napi_default, 0 },
        { "getNativeAbiVersion", 0, Method_GetNativeAbiVersion, 0, 0, 0, napi_default, 0 },
        { "getDiagnosticResultStructSize", 0, Method_GetDiagnosticResultStructSize, 0, 0, 0, napi_default, 0 },
        { "getVersion", 0, Method_GetVersion, 0, 0, 0, napi_default, 0 },
        { "isTensorAligned", 0, Method_IsTensorAligned, 0, 0, 0, napi_default, 0 },
        { "isVulkanAvailable", 0, Method_IsVulkanAvailable, 0, 0, 0, napi_default, 0 },
        { "quickProbe", 0, Method_QuickProbe, 0, 0, 0, napi_default, 0 },
        { "getConcurrencyMetrics", 0, Method_GetConcurrencyMetrics, 0, 0, 0, napi_default, 0 },
        { "resetConcurrencyMetrics", 0, Method_ResetConcurrencyMetrics, 0, 0, 0, napi_default, 0 },
        { "runDiagnostic", 0, Method_RunDiagnostic, 0, 0, 0, napi_default, 0 },
        { "matmulF32", 0, Method_MatmulF32, 0, 0, 0, napi_default, 0 },
        { "matmulF32WithTelemetry", 0, Method_MatmulF32WithTelemetry, 0, 0, 0, napi_default, 0 }
    };

    if (napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc) != napi_ok) {
        napi_throw_error(env, "MODULE_INIT_FAILED", "Failed to define properties on module exports");
        return NULL;
    }

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
