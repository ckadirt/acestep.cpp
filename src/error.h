#pragma once
// error.h: error reporting for engine code loaded as a shared library.
//
// Every load, allocation and compute path below the ABI reports failure by
// returning false/NULL and recording a code plus a message here, instead of
// calling exit(1). A CLI can print the message and exit; a long-lived host
// process can surface it and keep running.
//
// The record is thread-local: the server runs one worker thread today, but a
// process-wide slot would tear under any future concurrency and this costs
// nothing.
//
// Codes are the classes a caller can act on differently, nothing finer. OOM
// means retry smaller; BAD_MODEL means the file is wrong for this build;
// NO_BACKEND means fall back to another device; CANCELLED is a normal
// outcome, not a failure; INTERNAL means report it.

#include <cstdarg>
#include <cstdio>

enum AceErrorCode {
    ACE_OK = 0,
    ACE_ERR_OOM,         // device or host buffer allocation failed
    ACE_ERR_BAD_MODEL,   // missing/malformed tensor, unreadable GGUF, config mismatch
    ACE_ERR_NO_BACKEND,  // requested backend unavailable at runtime
    ACE_ERR_CANCELLED,   // cancel callback returned true
    ACE_ERR_INTERNAL,    // everything else
};

struct AceError {
    AceErrorCode code;
    char         msg[512];
};

static thread_local AceError g_ace_error = { ACE_OK, { 0 } };

// Record a failure. Also mirrors to stderr so CLI behaviour is unchanged from
// the exit(1) days: the operator still sees the message on the terminal.
static void ace_set_error(AceErrorCode code, const char * fmt, ...) {
    g_ace_error.code = code;

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_ace_error.msg, sizeof(g_ace_error.msg), fmt, ap);
    va_end(ap);

    fprintf(stderr, "%s\n", g_ace_error.msg);
    fflush(stderr);
}

static void ace_clear_error(void) {
    g_ace_error.code   = ACE_OK;
    g_ace_error.msg[0] = 0;
}

static AceErrorCode ace_error_code(void) {
    return g_ace_error.code;
}

static const char * ace_error_msg(void) {
    return g_ace_error.msg[0] ? g_ace_error.msg : "no error";
}

static const char * ace_error_name(AceErrorCode c) {
    switch (c) {
        case ACE_OK:              return "OK";
        case ACE_ERR_OOM:         return "OOM";
        case ACE_ERR_BAD_MODEL:   return "BAD_MODEL";
        case ACE_ERR_NO_BACKEND:  return "NO_BACKEND";
        case ACE_ERR_CANCELLED:   return "CANCELLED";
        case ACE_ERR_INTERNAL:    return "INTERNAL";
    }
    return "UNKNOWN";
}
