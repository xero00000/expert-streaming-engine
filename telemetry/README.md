# ESE community benchmark pipeline

ESE Studio's **Help improve ESE** option is disabled by default. When enabled, only a completed,
verified sweep is sanitized and queued for HTTPS upload. Local paths, prompts, responses, hostnames,
usernames, raw logs, and raw error messages are excluded in Studio before the payload leaves the PC.

Raw submissions are held in a private Cloudflare D1 database. The collector exposes no raw-data read
route. Its authenticated export route returns aggregate groups only, and suppresses groups containing
fewer than three submissions. The GitHub workflow publishes that aggregate export as the public list.

Disabling sharing stops future submissions and removes locally queued results.
It cannot retract an already accepted anonymous submission or an aggregate that
has already been published. Private rows are retained until the maintainer
deletes them; they are never exposed by the public export route.

## Local validation

```bash
cd telemetry/collector
npm ci
npm run check
```

The test suite runs in Cloudflare's Worker runtime with an isolated local D1
database. It covers request-size and schema gates, installation-ID hashing,
duplicate handling, export authentication, and the three-sample publication
threshold. The final dry run builds the deployable Worker without publishing
or touching the production database.

## Deploying the private collector

1. In `telemetry/collector`, install dependencies and create a D1 database named `ese-benchmarks`.
2. Copy `wrangler.jsonc.example` to `wrangler.jsonc` and set the private database ID.
3. Set strong Wrangler secrets named `INSTALLATION_HASH_SALT` and `EXPORT_TOKEN`.
4. Apply the remote migration, then deploy the Worker.
5. Set Studio's non-secret default collector URL in `studio/src-tauri/src/telemetry.rs`. The
   `ESE_BENCHMARK_COLLECTOR_URL` environment variable remains available for staging overrides.

For the public export workflow, configure repository secrets `ESE_BENCHMARK_EXPORT_URL` (the Worker
base URL) and `ESE_BENCHMARK_EXPORT_TOKEN`. Raw D1 data and these secrets must never be committed.
