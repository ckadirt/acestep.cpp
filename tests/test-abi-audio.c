/* test-abi-audio.c: audio in, through the C ABI.
 *
 * What a host needs from DIFFUSE once a request names audio by path:
 *   - the build says what it does with audio (cantor_engine_capabilities)
 *   - a cover reads src_audio_path and follows the source's length
 *   - an extend (repaint past the end) keeps the source's frames exactly:
 *     two seeds give byte-identical frames before the region and different
 *     frames inside it, which is the splice DECODE cannot do on its own
 *   - a task that needs a source and has none, or names a file that cannot be
 *     read, fails by the field's name
 *   - a cover paused, torn down and resumed matches the uninterrupted run,
 *     because the paths come back with the request
 *   - ENCODE turns a request's audio into latents once (a kept voice is the
 *     source, not a second encode), counts its tiles, and DIFFUSE from those
 *     latents gives the same cover, pauses without dropping them, and resumes
 *     to the same bytes; an extend from them still keeps the source's frames
 *
 * The source is made here: a short text2music render through the same ABI,
 * written as a 16-bit WAV. Usage: ./test-abi-audio --models <dir> [--tmp <dir>]
 */

#include "cantor_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void ok(const char * what) {
    fprintf(stderr, "[ABI-Audio] PASS: %s\n", what);
}

static void bad(const char * what) {
    fprintf(stderr, "[ABI-Audio] FAIL: %s (err=%d: %s)\n", what, (int) cantor_engine_last_error_code(),
            cantor_engine_last_error());
    g_fail = 1;
}

struct pauser {
    int after;
    int polls;
    int fired;
};

static int cancel_after(void * ud) {
    struct pauser * p = (struct pauser *) ud;
    if (p->fired) {
        return 0;
    }
    if (p->polls++ >= p->after) {
        p->fired = 1;
        return 1;
    }
    return 0;
}

static char * join(const char * dir, const char * name) {
    size_t n = strlen(dir) + strlen(name) + 2;
    char * p = (char *) malloc(n);
    snprintf(p, n, "%s/%s", dir, name);
    return p;
}

static void put16(FILE * f, uint16_t v) {
    fputc(v & 0xff, f);
    fputc(v >> 8, f);
}

static void put32(FILE * f, uint32_t v) {
    put16(f, (uint16_t) (v & 0xffff));
    put16(f, (uint16_t) (v >> 16));
}

/* planar stereo [L..., R...] -> 16-bit interleaved WAV */
static int write_wav(const char * path, const float * planar, int n, int rate) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return 0;
    }
    uint32_t data = (uint32_t) n * 4;
    fwrite("RIFF", 1, 4, f);
    put32(f, 36 + data);
    fwrite("WAVEfmt ", 1, 8, f);
    put32(f, 16);
    put16(f, 1);
    put16(f, 2);
    put32(f, (uint32_t) rate);
    put32(f, (uint32_t) rate * 4);
    put16(f, 4);
    put16(f, 16);
    fwrite("data", 1, 4, f);
    put32(f, data);
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 2; c++) {
            float v = planar[(size_t) c * n + i];
            v       = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
            put16(f, (uint16_t) (int16_t) (v * 32767.0f));
        }
    }
    fclose(f);
    return 1;
}

struct counted {
    int calls;
    int stage_ok;
    int last_i;
    int last_n;
};

static void count_progress(cantor_stage stage, int i, int n, void * ud) {
    struct counted * c = (struct counted *) ud;
    c->calls++;
    c->stage_ok = c->calls == 1 ? stage == CANTOR_STAGE_ENCODE : c->stage_ok && stage == CANTOR_STAGE_ENCODE;
    c->last_i   = i;
    c->last_n   = n;
}

static uint32_t u32_at(const uint8_t * p, size_t off) {
    uint32_t v;
    memcpy(&v, p + off, 4);
    return v;
}

/* run ENCODE; returns the encoded blob (caller frees) */
static uint8_t * encode(cantor_ctx * ctx, const char * json, size_t * len, struct counted * c) {
    uint8_t *     out = NULL;
    cantor_status st  = cantor_engine_run_stage(ctx, CANTOR_STAGE_ENCODE, (const uint8_t *) json, strlen(json), &out,
                                                len, c ? count_progress : NULL, NULL, c);
    if (st != CANTOR_DONE) {
        cantor_engine_free_blob(out);
        return NULL;
    }
    return out;
}

/* run DIFFUSE on a blob to completion; returns the latent blob (caller frees) */
static uint8_t * diffuse_blob(cantor_ctx * ctx, const uint8_t * in, size_t in_len, size_t * len) {
    uint8_t *     out = NULL;
    cantor_status st  = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, in, in_len, &out, len, NULL, NULL, NULL);
    if (st != CANTOR_DONE) {
        cantor_engine_free_blob(out);
        return NULL;
    }
    return out;
}

/* run DIFFUSE to completion; returns the latent blob (caller frees) */
static uint8_t * diffuse(cantor_ctx * ctx, const char * json, size_t * len) {
    uint8_t *     out = NULL;
    cantor_status st  = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, (const uint8_t *) json, strlen(json), &out,
                                                len, NULL, NULL, NULL);
    if (st != CANTOR_DONE) {
        cantor_engine_free_blob(out);
        return NULL;
    }
    return out;
}

#define SOURCE_SECONDS 6
#define EXTEND_SECONDS 3
#define FRAMES_PER_SECOND 25

int main(int argc, char ** argv) {
    const char * models = NULL;
    const char * tmp    = "/tmp";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models = argv[++i];
        } else if (!strcmp(argv[i], "--tmp") && i + 1 < argc) {
            tmp = argv[++i];
        }
    }
    if (!models) {
        fprintf(stderr, "Usage: %s --models <dir> [--tmp <dir>]\n", argv[0]);
        return 2;
    }

    /* ---- 0. what the build says it does with audio ---- */
    const char * caps = cantor_engine_capabilities();
    fprintf(stderr, "[ABI-Audio] capabilities: %s\n", caps ? caps : "(null)");
    if (!caps || !strstr(caps, "cover") || !strstr(caps, "extend") || !strstr(caps, "timbre")) {
        bad("capabilities should name cover, extend and timbre");
    } else {
        ok("capabilities declared");
    }

    char * dit   = join(models, "acestep-v15-turbo-Q4_K_M.gguf");
    char * embed = join(models, "Qwen3-Embedding-0.6B-Q8_0.gguf");
    char * vae   = join(models, "vae-BF16.gguf");
    char * src   = join(tmp, "cantor-abi-audio-source.wav");

    cantor_component comps[3] = {
        { "dit", dit },
        { "embed", embed },
        { "vae", vae },
    };
    cantor_load_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.vae_chunk = 256;

    cantor_ctx * ctx = cantor_engine_load(comps, 3, &opts);
    if (!ctx) {
        bad("engine_load");
        return 1;
    }

    /* ---- 1. make a source: a short render, written as WAV ---- */
    {
        char req[512];
        snprintf(req, sizeof(req),
                 "{\"caption\":\"Warm piano chords with soft brushed drums\",\"lyrics\":\"[Instrumental]\","
                 "\"bpm\":100,\"duration\":%d,\"keyscale\":\"C major\",\"timesignature\":\"4\","
                 "\"vocal_language\":\"unknown\",\"seed\":7,\"inference_steps\":4,\"shift\":3.0}",
                 SOURCE_SECONDS);
        size_t    len = 0;
        uint8_t * lat = diffuse(ctx, req, &len);
        uint8_t * unused     = NULL;
        size_t    unused_len = 0;
        if (!lat || cantor_engine_run_stage(ctx, CANTOR_STAGE_DECODE, lat, len, &unused, &unused_len, NULL, NULL,
                                            NULL) != CANTOR_DONE) {
            bad("render the source");
            return 1;
        }
        cantor_engine_free_blob(lat);
        int           n = 0, rate = 0;
        const float * pcm = cantor_engine_audio(ctx, &n, &rate);
        if (!pcm || !write_wav(src, pcm, n, rate)) {
            bad("write the source WAV");
            return 1;
        }
        fprintf(stderr, "[ABI-Audio] source: %s, %.2fs\n", src, (double) n / rate);
    }

    /* ---- 2. cover from a path follows the source's length ---- */
    char cover_req[1024];
    snprintf(cover_req, sizeof(cover_req),
             "{\"task_type\":\"cover\",\"caption\":\"Solo acoustic guitar\",\"lyrics\":\"[Instrumental]\","
             "\"src_audio_path\":\"%s\",\"ref_audio_path\":\"%s\",\"audio_cover_strength\":0.6,"
             "\"seed\":11,\"inference_steps\":4,\"shift\":3.0}",
             src, src);
    size_t    cover_len = 0;
    uint8_t * cover     = diffuse(ctx, cover_req, &cover_len);
    if (!cover) {
        bad("cover DIFFUSE");
    } else {
        size_t T = cover_len / 256;
        fprintf(stderr, "[ABI-Audio] cover latent: T=%zu\n", T);
        if (T < (size_t) (SOURCE_SECONDS * FRAMES_PER_SECOND) - 2 || T > (size_t) (SOURCE_SECONDS * FRAMES_PER_SECOND) + 2) {
            bad("cover length should follow the source");
        } else {
            ok("cover from src_audio_path follows the source's length");
        }
    }

    /* ---- 3. extend keeps the source's frames, whatever the seed ---- */
    {
        uint8_t * runs[2]    = { NULL, NULL };
        size_t    lens[2]    = { 0, 0 };
        int       seeds[2]   = { 21, 22 };
        for (int k = 0; k < 2; k++) {
            char req[1024];
            snprintf(req, sizeof(req),
                     "{\"task_type\":\"repaint\",\"caption\":\"Warm piano chords with soft brushed drums\","
                     "\"lyrics\":\"[Instrumental]\",\"src_audio_path\":\"%s\",\"repainting_start\":%d,"
                     "\"repainting_end\":%d,\"seed\":%d,\"inference_steps\":4,\"shift\":3.0}",
                     src, SOURCE_SECONDS, SOURCE_SECONDS + EXTEND_SECONDS, seeds[k]);
            runs[k] = diffuse(ctx, req, &lens[k]);
        }
        if (!runs[0] || !runs[1]) {
            bad("extend DIFFUSE");
        } else {
            size_t T    = lens[0] / 256;
            size_t keep = (size_t) (SOURCE_SECONDS * FRAMES_PER_SECOND) - 1;
            fprintf(stderr, "[ABI-Audio] extend latent: T=%zu (source ~%d frames)\n", T,
                    SOURCE_SECONDS * FRAMES_PER_SECOND);
            if (lens[0] != lens[1] || T < (size_t) ((SOURCE_SECONDS + EXTEND_SECONDS) * FRAMES_PER_SECOND) - 2) {
                bad("extend should cover the source plus the added time");
            } else if (memcmp(runs[0], runs[1], keep * 256) != 0) {
                bad("the source's frames differ between seeds: the splice did not run");
            } else if (memcmp(runs[0] + (T - 10) * 256, runs[1] + (T - 10) * 256, 10 * 256) == 0) {
                bad("the added frames are identical between seeds: nothing was generated");
            } else {
                ok("extend keeps the source's frames exactly and generates the rest");
            }
        }
        cantor_engine_free_blob(runs[0]);
        cantor_engine_free_blob(runs[1]);
    }

    /* ---- 4. refusals name the field ---- */
    {
        const char * no_src = "{\"task_type\":\"cover\",\"caption\":\"x\",\"lyrics\":\"[Instrumental]\"}";
        size_t       len    = 0;
        uint8_t *    out    = diffuse(ctx, no_src, &len);
        if (out || !strstr(cantor_engine_last_error(), "src_audio_path")) {
            bad("cover without a source should fail naming src_audio_path");
        } else {
            ok("cover without a source is refused by name");
        }
        cantor_engine_free_blob(out);

        const char * missing = "{\"task_type\":\"cover\",\"caption\":\"x\",\"lyrics\":\"[Instrumental]\","
                               "\"src_audio_path\":\"/nonexistent/tune.wav\"}";
        out = diffuse(ctx, missing, &len);
        if (out || !strstr(cantor_engine_last_error(), "src_audio_path")) {
            bad("an unreadable source should fail naming src_audio_path");
        } else {
            ok("an unreadable source is refused by name");
        }
        cantor_engine_free_blob(out);
    }
    cantor_engine_free(ctx);

    /* ---- 5. a cover paused, torn down and resumed matches the straight run ---- */
    if (cover) {
        struct pauser p      = { 2, 0, 0 };
        uint8_t *     paused = NULL;
        size_t        plen   = 0;
        ctx                  = cantor_engine_load(comps, 3, &opts);
        cantor_status st     = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, (const uint8_t *) cover_req,
                                                       strlen(cover_req), &paused, &plen, NULL, cancel_after, &p);
        cantor_engine_free(ctx);
        if (st != CANTOR_PAUSED || !paused) {
            bad("cover DIFFUSE did not pause");
        } else {
            ctx               = cantor_engine_load(comps, 3, &opts);
            uint8_t * resumed = NULL;
            size_t    rlen    = 0;
            st = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, paused, plen, &resumed, &rlen, NULL, NULL, NULL);
            if (st != CANTOR_DONE || rlen != cover_len || memcmp(resumed, cover, rlen) != 0) {
                bad("a resumed cover should match the uninterrupted run");
            } else {
                ok("a cover resumes across a teardown, byte-identical");
            }
            cantor_engine_free_blob(resumed);
            cantor_engine_free(ctx);
        }
        cantor_engine_free_blob(paused);
    }

    /* ---- 6. ENCODE: once, counted, and DIFFUSE from its latents ---- */
    ctx = cantor_engine_load(comps, 3, &opts);
    if (!(cantor_engine_stages() & (1u << CANTOR_STAGE_ENCODE))) {
        bad("the stage mask should offer ENCODE");
    } else {
        ok("the stage mask offers ENCODE");
    }
    {
        struct counted c   = { 0, 0, 0, 0 };
        size_t         len = 0;
        uint8_t *      enc = encode(ctx, cover_req, &len, &c);
        if (!enc || len < 24 || memcmp(enc, "ACEENC01", 8) != 0) {
            bad("ENCODE a cover");
        } else {
            uint32_t T_src = u32_at(enc, 12), T_ref = u32_at(enc, 16), ref_is_src = u32_at(enc, 20);
            fprintf(stderr, "[ABI-Audio] encoded: T_src=%u T_ref=%u ref_is_src=%u, %d progress calls (%d/%d)\n",
                    T_src, T_ref, ref_is_src, c.calls, c.last_i, c.last_n);
            if (ref_is_src != 1 || T_ref != 0 || len != 24 + u32_at(enc, 8) + (size_t) T_src * 256) {
                bad("a kept voice should be the source's latents, not a second encode");
            } else {
                ok("a kept voice is encoded once");
            }
            if (c.calls < 2 || !c.stage_ok || c.last_i != c.last_n) {
                bad("ENCODE should count its tiles as the ENCODE stage, to the end");
            } else {
                ok("ENCODE counts its tiles");
            }

            size_t    dlen = 0;
            uint8_t * from = diffuse_blob(ctx, enc, len, &dlen);
            if (!from || !cover || dlen != cover_len || memcmp(from, cover, dlen) != 0) {
                bad("DIFFUSE from ENCODE's latents should give the same cover");
            } else {
                ok("DIFFUSE from ENCODE's latents gives the same cover");
            }
            cantor_engine_free_blob(from);
            cantor_engine_free(ctx);

            /* paused from latents, torn down, resumed: no re-encode, same bytes */
            struct pauser p      = { 2, 0, 0 };
            uint8_t *     paused = NULL;
            size_t        plen   = 0;
            ctx                  = cantor_engine_load(comps, 3, &opts);
            cantor_status st =
                cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, enc, len, &paused, &plen, NULL, cancel_after, &p);
            cantor_engine_free(ctx);
            if (st != CANTOR_PAUSED || !paused || memcmp(paused, "ACEDIF02", 8) != 0) {
                bad("DIFFUSE from latents should pause into a blob that keeps them");
            } else {
                ctx               = cantor_engine_load(comps, 3, &opts);
                size_t    rlen    = 0;
                uint8_t * resumed = diffuse_blob(ctx, paused, plen, &rlen);
                if (!resumed || !cover || rlen != cover_len || memcmp(resumed, cover, rlen) != 0) {
                    bad("a cover resumed from latents should match the uninterrupted run");
                } else {
                    ok("a cover resumes from its latents across a teardown, byte-identical");
                }
                cantor_engine_free_blob(resumed);
            }
            cantor_engine_free_blob(paused);
        }
        cantor_engine_free_blob(enc);
    }
    {
        /* an extend from latents still keeps the source's frames */
        uint8_t * runs[2]  = { NULL, NULL };
        size_t    lens[2]  = { 0, 0 };
        int       seeds[2] = { 21, 22 };
        for (int k = 0; k < 2; k++) {
            char req[1024];
            snprintf(req, sizeof(req),
                     "{\"task_type\":\"repaint\",\"caption\":\"Warm piano chords with soft brushed drums\","
                     "\"lyrics\":\"[Instrumental]\",\"src_audio_path\":\"%s\",\"repainting_start\":%d,"
                     "\"repainting_end\":%d,\"seed\":%d,\"inference_steps\":4,\"shift\":3.0}",
                     src, SOURCE_SECONDS, SOURCE_SECONDS + EXTEND_SECONDS, seeds[k]);
            size_t    elen = 0;
            uint8_t * enc  = encode(ctx, req, &elen, NULL);
            runs[k]        = enc ? diffuse_blob(ctx, enc, elen, &lens[k]) : NULL;
            cantor_engine_free_blob(enc);
        }
        size_t keep = (size_t) (SOURCE_SECONDS * FRAMES_PER_SECOND) - 1;
        size_t T    = lens[0] / 256;
        if (!runs[0] || !runs[1] || lens[0] != lens[1] ||
            T < (size_t) ((SOURCE_SECONDS + EXTEND_SECONDS) * FRAMES_PER_SECOND) - 2) {
            bad("extend from latents");
        } else if (memcmp(runs[0], runs[1], keep * 256) != 0) {
            bad("an extend from latents changed the source's frames");
        } else if (memcmp(runs[0] + (T - 10) * 256, runs[1] + (T - 10) * 256, 10 * 256) == 0) {
            bad("an extend from latents generated nothing");
        } else {
            ok("an extend from latents keeps the source's frames and generates the rest");
        }
        cantor_engine_free_blob(runs[0]);
        cantor_engine_free_blob(runs[1]);
    }
    {
        /* small tiles: the tiled path counts every tile and stops between them */
        cantor_load_opts small = opts;
        small.vae_chunk        = 32;
        small.vae_overlap      = 8;
        cantor_ctx *   tiled   = cantor_engine_load(comps, 3, &small);
        struct counted c       = { 0, 0, 0, 0 };
        size_t         len     = 0;
        uint8_t *      enc     = tiled ? encode(tiled, cover_req, &len, &c) : NULL;
        fprintf(stderr, "[ABI-Audio] tiled encode: %d progress calls (%d/%d)\n", c.calls, c.last_i, c.last_n);
        if (!enc || c.last_n < 3 || c.calls != c.last_n + 1 || c.last_i != c.last_n || !c.stage_ok) {
            bad("a tiled ENCODE should report 0..n tiles as the ENCODE stage");
        } else {
            ok("a tiled ENCODE reports every tile");
        }
        cantor_engine_free_blob(enc);

        struct pauser p      = { 1, 0, 0 };
        uint8_t *     out    = NULL;
        size_t        olen   = 0;
        cantor_status st     = cantor_engine_run_stage(tiled, CANTOR_STAGE_ENCODE, (const uint8_t *) cover_req,
                                                       strlen(cover_req), &out, &olen, NULL, cancel_after, &p);
        if (st != CANTOR_PAUSED || olen != strlen(cover_req) || memcmp(out, cover_req, olen) != 0) {
            fprintf(stderr, "[ABI-Audio] stopped ENCODE: status %d, %zu bytes, %d polls\n", (int) st, olen, p.polls);
            bad("a stopped ENCODE should pause with the request to start over from");
        } else {
            ok("a stopped ENCODE pauses with its request");
        }
        cantor_engine_free_blob(out);
        cantor_engine_free(tiled);
    }
    {
        const char * none = "{\"task_type\":\"cover\",\"caption\":\"x\",\"lyrics\":\"[Instrumental]\"}";
        size_t       len  = 0;
        uint8_t *    out  = encode(ctx, none, &len, NULL);
        if (out || !strstr(cantor_engine_last_error(), "src_audio_path")) {
            bad("ENCODE with no audio should fail naming src_audio_path");
        } else {
            ok("ENCODE with no audio is refused by name");
        }
        cantor_engine_free_blob(out);
    }
    cantor_engine_free(ctx);

    cantor_engine_free_blob(cover);
    remove(src);
    free(src);
    free(dit);
    free(embed);
    free(vae);

    fprintf(stderr, "\n[ABI-Audio] %s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail;
}
