// uriel0.h  (generated via cbindgen)
#ifdef __cplusplus
extern "C" {
#endif
typedef struct uriel0_ctx uriel0_ctx;

typedef struct { const char* ptr; size_t len; } uriel0_str;

typedef struct { float probs[4]; uint8_t cls; float conf; } uriel0_route;

typedef struct {
  uriel0_str text;        // UTF-8 bytes, URIEL alloc; call uriel0_free_blob
  uriel0_str source;      // "db:def", "rows", "pfc", etc.
  float confidence;       // optional combined conf
  uriel0_route route;     // router snapshot used for this answer
} uriel0_answer;

uint32_t uriel0_abi_version(void); // e.g., 0x00010000

int  uriel0_new(uriel0_ctx** out_ctx, const char* db_path /*nullable*/);
void uriel0_free(uriel0_ctx*);

int  uriel0_load_model(uriel0_ctx*, const char* checkpoint_path);
int  uriel0_load_router(uriel0_ctx*, const char* record_path, const char* calib_json_path);
int  uriel0_set_context(uriel0_ctx*, const char* ctx_name);

/* Batch, to keep FFI overhead tiny */
int  uriel0_route_batch(uriel0_ctx*, const uriel0_str* prompts, size_t n,
                        uriel0_route* out_routes);             // out_routes[n]
int  uriel0_ingest_batch(uriel0_ctx*, const uriel0_str* lines, size_t n,
                         uint64_t ttl_secs, const char* source_tag);

/* Synchronous single prompt (mesh can still call in parallel from many threads) */
int  uriel0_chat(uriel0_ctx*, uriel0_str prompt, uriel0_answer* out);
int  uriel0_read_fact(uriel0_ctx*, uriel0_str key, uriel0_str* out_value);
int  uriel0_write_fact(uriel0_ctx*, uriel0_str key, uriel0_str value); // optional

/* Memory management for returned blobs */
void uriel0_free_blob(const void* ptr, size_t len);
#ifdef __cplusplus
}
#endif
