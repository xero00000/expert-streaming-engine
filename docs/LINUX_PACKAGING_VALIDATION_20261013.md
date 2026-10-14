# Linux candidate packaging validation — 2026-10-13

The six launcher/Studio version surfaces now agree on `0.2.1`. Local unsigned
DEB, RPM and AppImage candidates build and execute on Nobara Linux 44. These
packages bundle the CPU native runtime. This report establishes local
packaging and runtime behavior, not release readiness or compatibility with
every Linux distribution.

Evidence: `/home/xero/Desktop/test/results/ese-studio-packaging-20261013`.

## Repairs

- CPU runtime staging now sets ELF search paths to `$ORIGIN`, using its bundled
  native libraries and system compiler libraries. The previous staging retained
  absolute build and Conda paths. CUDA staging removes both source and selected
  build directories while retaining external vendor runtime locations.
  Two permanent staging cases fail against the old script and pass now.
- Release asset collection selects exactly one bundle of the requested version
  and its matching signature. It rejects missing, empty or ambiguous inputs and
  a nonempty output directory. Older cached installers cannot be renamed as
  the current release. Eleven cases cover Linux and Windows selection; they do
  not establish Windows installer execution or cryptographic verification.
- The AppImage build failed because linuxdeploy's bundled `strip` rejected
  `.relr.dyn` sections in the host libraries. Its documented `NO_STRIP=1` option
  permits bundling without that transformation. The release workflow now sets
  this option, and the resulting AppImage passes the checks below. The option
  is documented by the [linuxdeploy maintainer](https://github.com/linuxdeploy/linuxdeploy/issues/72).
- Both release package jobs now depend on a complete native CPU regression
  job. Publication depends on those package jobs. The dependency graph is
  checked locally; the updated GitHub workflow has not been executed.

## Executed checks

Locked frontend dependency installation and production build pass. Rust
formatting, Clippy with warnings denied, and all 27 Rust tests pass. All 19
Studio tests pass. Python unittest passes 98 tests; pytest passes 164 tests and
122 subtests. The Python suites overlap.

Each of the three candidate packages passes:

1. version, license-notice and desktop-file checks;
2. packaged launcher version and doctor, and native server version checks;
3. ELF dependency checks with no build or Conda fallback;
4. actual HTTP inference using the retained synthetic three-layer fixture;
5. exact saved teacher output and all 32 probabilities at context 512, after
   shrinking to 256, and after restoring 512;
6. loaded-library maps proving that the packaged native libraries are used;
7. Studio startup for at least 12 seconds on an isolated virtual display.

Screenshots show first-run onboarding with benchmark sharing off and the
bundled runtime ready. GUI checks use isolated configuration/data/cache
directories. Packages were extracted for validation; no system installation
or publication was performed. The AppImage GUI check runs its actual entry
point with extraction, and its native/API checks use the extracted payload.

## Scope remaining open

These are unsigned local candidates. Ubuntu CI execution, Windows CPU/CUDA
installers, signed updater artifacts and publication remain unqualified.
The native suites still retain one initialized DeepSeek block failure.
Original Qwen numerical parity, full 64K execution, concurrent serving and
matched Strata evidence remain open. Normal Qwen loading stays gated, and
the release remains unready.
