// test-dit-resume.cpp: prove a paused DiT run resumes to the same latents.
//
// Milestone 7's central claim is that cancelling mid-schedule and continuing
// costs only the steps not yet run, and changes nothing about the result. A
// resume that drifts is worse than no resume at all, because the user cannot
// see that it happened.
//
// Scenario:
//   1. run the schedule straight through, keep the latents
//   2. run again with a cancel callback that fires once at step k, then
//      resume, and compare the latents byte for byte
//   3. check a resume is refused on a stateful solver, rather than silently
//      producing a different track
//
// Usage:
//   ./test-dit-resume --models <dir> [--steps N] [--pause-at K] [--duration S]

#include "model-registry.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "request.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Cancel callback that fires exactly once, when the DiT has completed
// `fire_after` steps. The sampler polls this at the top of every step, so
// returning true on the (fire_after+1)-th poll pauses entering step fire_after.
struct PauseAt {
    int  fire_after;
    int  polls;
    bool fired;
};

static bool pause_cb(void * ud) {
    PauseAt * p = (PauseAt *) ud;
    if (p->fired) {
        return false;  // let the resumed run continue to the end
    }
    if (p->polls++ >= p->fire_after) {
        p->fired = true;
        return true;
    }
    return false;
}

static AceRequest make_request(const char * synth_model, int steps, float duration, const char * solver) {
    AceRequest r;
    request_init(&r);
    r.caption         = "Ambient electronic soundscape with warm analog pads";
    r.lyrics          = "[Instrumental]";
    r.bpm             = 90;
    r.duration        = duration;
    r.keyscale        = "C minor";
    r.timesignature   = "4";
    r.vocal_language  = "en";
    r.seed            = 42;
    r.inference_steps = steps;
    r.guidance_scale  = 1.0f;
    r.shift           = 3.0f;
    r.solver          = solver;
    r.synth_model     = synth_model;
    return r;
}

int main(int argc, char ** argv) {
    const char * models_dir = nullptr;
    int          steps      = 8;
    int          pause_at   = 5;
    float        duration   = 12.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pause-at") && i + 1 < argc) {
            pause_at = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--duration") && i + 1 < argc) {
            duration = (float) atof(argv[++i]);
        }
    }
    if (!models_dir) {
        fprintf(stderr, "Usage: %s --models <dir> [--steps N] [--pause-at K] [--duration S]\n", argv[0]);
        return 2;
    }

    ModelRegistry reg;
    if (!registry_scan(&reg, models_dir)) {
        fprintf(stderr, "[Test] FAIL: cannot scan %s\n", models_dir);
        return 1;
    }
    if (reg.dit.empty() || reg.text_enc.empty() || reg.vae.empty()) {
        fprintf(stderr, "[Test] FAIL: registry missing dit/text-enc/vae in %s\n", models_dir);
        return 1;
    }
    const char * dit_path  = reg.dit[0].path.c_str();
    const char * text_path = reg.text_enc[0].path.c_str();
    const char * vae_path  = reg.vae[0].path.c_str();
    fprintf(stderr, "[Test] dit=%s\n[Test] text-enc=%s\n[Test] vae=%s\n", dit_path, text_path, vae_path);

    ModelStore * store = store_create(EVICT_STRICT);

    AceSynthParams params;
    ace_synth_default_params(&params);
    params.text_encoder_path = text_path;
    params.dit_path          = dit_path;
    params.vae_path          = vae_path;

    AceSynth * ctx = ace_synth_load(store, &params);
    if (!ctx) {
        fprintf(stderr, "[Test] FAIL: ace_synth_load\n");
        store_free(store);
        return 1;
    }

    int rc = 0;

    // ---- 1. uninterrupted reference ----
    fprintf(stderr, "\n[Test] === reference run (%d steps, no pause) ===\n", steps);
    std::vector<float> reference;
    {
        AceRequest    r   = make_request(dit_path, steps, duration, "euler");
        AceSynthJob * job = ace_synth_job_run_dit(ctx, &r, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1);
        if (!job) {
            fprintf(stderr, "[Test] FAIL: reference run_dit returned NULL\n");
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        if (ace_synth_job_is_paused(job)) {
            fprintf(stderr, "[Test] FAIL: reference run paused without a cancel callback\n");
            ace_synth_job_free(job);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        int           T   = 0;
        const float * lat = ace_synth_job_get_latent(job, 0, &T);
        reference.assign(lat, lat + (size_t) T * 64);
        fprintf(stderr, "[Test] reference latents: T=%d (%zu floats)\n", T, reference.size());
        ace_synth_job_free(job);
    }

    // ---- 2. pause at K, then resume ----
    fprintf(stderr, "\n[Test] === paused run (pause entering step %d) ===\n", pause_at);
    {
        AceRequest r  = make_request(dit_path, steps, duration, "euler");
        PauseAt    pa = { pause_at, 0, false };

        AceSynthJob * job =
            ace_synth_job_run_dit(ctx, &r, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1, pause_cb, &pa);
        if (!job) {
            fprintf(stderr, "[Test] FAIL: paused run_dit returned NULL (pause must not destroy the job)\n");
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        if (!ace_synth_job_is_paused(job)) {
            fprintf(stderr, "[Test] FAIL: expected a paused job, got a completed one\n");
            ace_synth_job_free(job);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        int total = 0;
        int next  = ace_synth_job_progress(job, &total);
        fprintf(stderr, "[Test] paused at step %d/%d\n", next, total);
        if (next != pause_at) {
            fprintf(stderr, "[Test] FAIL: expected pause at %d, got %d\n", pause_at, next);
            ace_synth_job_free(job);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }

        fprintf(stderr, "[Test] resuming...\n");
        int resume_rc = ace_synth_job_resume_dit(ctx, job, pause_cb, &pa);
        if (resume_rc != 0) {
            fprintf(stderr, "[Test] FAIL: resume returned %d, expected 0 (completed)\n", resume_rc);
            ace_synth_job_free(job);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        if (ace_synth_job_is_paused(job)) {
            fprintf(stderr, "[Test] FAIL: job still marked paused after completing\n");
            ace_synth_job_free(job);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }

        int           T   = 0;
        const float * lat = ace_synth_job_get_latent(job, 0, &T);
        if ((size_t) T * 64 != reference.size()) {
            fprintf(stderr, "[Test] FAIL: resumed T=%d does not match reference\n", T);
            ace_synth_job_free(job);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }

        size_t diffs     = 0;
        float  max_delta = 0.0f;
        for (size_t i = 0; i < reference.size(); i++) {
            if (lat[i] != reference[i]) {
                diffs++;
                float d = lat[i] - reference[i];
                if (d < 0) {
                    d = -d;
                }
                if (d > max_delta) {
                    max_delta = d;
                }
            }
        }
        if (diffs == 0) {
            fprintf(stderr, "[Test] PASS: resumed latents are byte-identical to the reference\n");
        } else {
            fprintf(stderr, "[Test] FAIL: %zu/%zu floats differ, max delta %.9g\n", diffs, reference.size(),
                    (double) max_delta);
            rc = 1;
        }
        ace_synth_job_free(job);
    }

    // ---- 3. stateful solver must refuse to resume ----
    fprintf(stderr, "\n[Test] === resume refusal on a stateful solver (dpm3m) ===\n");
    {
        AceRequest r  = make_request(dit_path, steps, duration, "dpm3m");
        PauseAt    pa = { pause_at, 0, false };

        AceSynthJob * job =
            ace_synth_job_run_dit(ctx, &r, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1, pause_cb, &pa);
        if (!job || !ace_synth_job_is_paused(job)) {
            fprintf(stderr, "[Test] FAIL: expected a paused dpm3m job\n");
            if (job) {
                ace_synth_job_free(job);
            }
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        int resume_rc = ace_synth_job_resume_dit(ctx, job, pause_cb, &pa);
        if (resume_rc == -1) {
            fprintf(stderr, "[Test] PASS: resume refused on a stateful solver\n");
        } else {
            fprintf(stderr, "[Test] FAIL: dpm3m resume returned %d, expected -1 (refusal)\n", resume_rc);
            rc = 1;
        }
        ace_synth_job_free(job);
    }

    ace_synth_free(ctx);
    store_free(store);
    fprintf(stderr, "\n[Test] %s\n", rc == 0 ? "ALL PASS" : "FAILURES");
    return rc;
}
