// Secret names cannot be inferred by `wrangler types` because their values and names are not
// stored in wrangler.jsonc. Non-secret bindings are generated in worker-configuration.d.ts.
interface Env {
  INSTALLATION_HASH_SALT: string;
  EXPORT_TOKEN: string;
}
