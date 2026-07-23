/* test-backend-guard.c: prove a resume is refused across backends.
 *
 * The paused DIFFUSE blob stamps the backend it was produced on. Resuming
 * elsewhere re-derives the conditioning through different kernels, so the
 * remaining trajectory diverges - a different track, not a broken one, which
 * is the kind of wrongness nobody can see. The engine must refuse.
 *
 * Two modes, so the two halves can run as separate processes with different
 * GGML_BACKEND values:
 *   --save <file>     pause a diffusion early, write the blob
 *   --resume <file>   read the blob, attempt to continue
 *
 * Expected: same backend for both -> resume proceeds.
 *           different backend      -> CANTOR_ERR naming both.
 */

#include "cantor_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fire_now(void * ud) {
    int * n = (int *) ud;
    return (*n)++ >= 2; /* pause a couple of steps in */
}

static char * join(const char * d, const char * f) {
    size_t n = strlen(d) + strlen(f) + 2;
    char * p = malloc(n);
    snprintf(p, n, "%s/%s", d, f);
    return p;
}

static const char * REQ =
    "{\"caption\":\"Ambient pads\",\"lyrics\":\"[Instrumental]\",\"duration\":12,"
    "\"seed\":42,\"inference_steps\":8,\"guidance_scale\":1.0,\"shift\":3.0,\"solver\":\"euler\"}";

int main(int argc, char ** argv) {
    const char * models = NULL;
    const char * mode   = NULL;
    const char * file   = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models = argv[++i];
        } else if (!strcmp(argv[i], "--save") && i + 1 < argc) {
            mode = "save"; file = argv[++i];
        } else if (!strcmp(argv[i], "--resume") && i + 1 < argc) {
            mode = "resume"; file = argv[++i];
        }
    }
    if (!models || !mode) {
        fprintf(stderr, "Usage: %s --models <dir> (--save|--resume) <file>\n", argv[0]);
        return 2;
    }

    char * dit   = join(models, "acestep-v15-turbo-Q4_K_M.gguf");
    char * embed = join(models, "Qwen3-Embedding-0.6B-Q8_0.gguf");
    char * vae   = join(models, "vae-BF16.gguf");
    cantor_component comps[3] = { { "dit", dit }, { "embed", embed }, { "vae", vae } };
    cantor_load_opts opts;
    memset(&opts, 0, sizeof(opts));

    cantor_ctx * ctx = cantor_engine_load(comps, 3, &opts);
    if (!ctx) {
        fprintf(stderr, "[Guard] load failed: %s\n", cantor_engine_last_error());
        return 1;
    }

    int rc = 0;
    if (!strcmp(mode, "save")) {
        int polls = 0;
        uint8_t * blob = NULL; size_t bl = 0;
        cantor_status st = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, (const uint8_t *) REQ, strlen(REQ),
                                                   &blob, &bl, NULL, fire_now, &polls);
        if (st != CANTOR_PAUSED || !blob) {
            fprintf(stderr, "[Guard] FAIL: expected a pause, got %d\n", (int) st);
            rc = 1;
        } else {
            FILE * f = fopen(file, "wb");
            fwrite(blob, 1, bl, f);
            fclose(f);
            fprintf(stderr, "[Guard] saved %zu byte blob to %s\n", bl, file);
            cantor_engine_free_blob(blob);
        }
    } else {
        FILE * f = fopen(file, "rb");
        if (!f) { fprintf(stderr, "[Guard] cannot open %s\n", file); return 1; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t * in = malloc((size_t) n);
        if (fread(in, 1, (size_t) n, f) != (size_t) n) { fprintf(stderr, "[Guard] short read\n"); return 1; }
        fclose(f);

        uint8_t * out = NULL; size_t ol = 0;
        cantor_status st = cantor_engine_run_stage(ctx, CANTOR_STAGE_DIFFUSE, in, (size_t) n, &out, &ol, NULL, NULL,
                                                   NULL);
        if (st == CANTOR_ERR) {
            fprintf(stderr, "[Guard] REFUSED: %s\n", cantor_engine_last_error());
            printf("REFUSED\n");
        } else {
            fprintf(stderr, "[Guard] ACCEPTED (status=%d)\n", (int) st);
            printf("ACCEPTED\n");
            cantor_engine_free_blob(out);
        }
        free(in);
    }

    cantor_engine_free(ctx);
    free(dit); free(embed); free(vae);
    return rc;
}
