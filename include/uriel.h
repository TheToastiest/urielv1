#include <stdint.h>

#ifndef URIEL_H
#define URIEL_H

#pragma once

typedef enum UrielEventKind {
  Episode = 0,
  Fact = 1,
} UrielEventKind;

typedef enum UrielStatus {
  URIEL_OK = 0,
  URIEL_ERR_NULL = 1,
  URIEL_ERR_UTF8 = 2,
  URIEL_ERR_INTERNAL = 3,
} UrielStatus;

typedef struct Inner Inner;

typedef struct UrielHandle {
  struct Inner *_private;
} UrielHandle;

typedef void (*UrielEventCallback)(void*, enum UrielEventKind, const char*);

/**
 * Free a `char*` allocated by Rust (`uriel_ask_json` / future allocators).
 */
void uriel_string_free(char *p);

/**
 * Initialize and return a UrielHandle**. Returns URIEL_OK on success.
 */
enum UrielStatus uriel_init(struct UrielHandle **out);

/**
 * Destroy the handle and all internal resources.
 */
void uriel_free(struct UrielHandle *handle);

/**
 * Set the event callback. Pass `NULL` to clear.
 */
enum UrielStatus uriel_set_event_callback(UrielEventCallback cb, void *user_data);

/**
 * Optional: Connect RiftNet (stub for now — records host/port).
 */
enum UrielStatus uriel_riftnet_connect(struct UrielHandle *handle, const char *host, uint16_t port);

/**
 * Ask the runtime with a small JSON envelope; returns malloc'd JSON string owned by Rust.
 * C/C++ must free via `uriel_string_free`.
 */
enum UrielStatus uriel_ask_json(struct UrielHandle *handle,
                                const char *prompt_utf8,
                                char **out_json);

uint32_t uriel_abi_version_major(void);

uint32_t uriel_abi_version_minor(void);

#endif /* URIEL_H */
