# ESE GitHub presentation

## Message

**Bigger AI. The PC you already own.**

Introduce ESE as a desktop app for finding models and chatting locally, then
explain how supported sparse models can use SSD, RAM and NVIDIA GPUs together.
Keep model compatibility, memory-fit estimates and measured performance distinct.

## Repository About copy

Bigger local AI on the PC you already own. Find models, chat locally, and run
supported sparse models across SSD, RAM and NVIDIA GPUs with ESE Studio.

Suggested topics: `local-ai`, `local-llm`, `desktop-app`, `gguf`, `inference`,
`mixture-of-experts`, `expert-streaming`, `multi-gpu`, `cuda`, `linux`, `windows`,
`tauri`, `llama-cpp`.

Keep the existing releases homepage so visitors can inspect actual published
versions. Repository naming and the app icon stay consistent with ESE Studio.

## Visual assets

- [Wide README banner](brand/ese-hero.svg): editable SVG, 1280 × 640.
- [Narrow README banner](brand/ese-hero-mobile.svg): editable SVG, 760 × 760.
- [Social preview](brand/ese-social-preview.png): 1280 × 640 PNG, rendered from
  the wide SVG for the repository's social preview setting.

The banners use ESE's existing block E, charcoal surfaces and a mint accent.
They illustrate the SSD → RAM → GPU idea for supported sparse models. They are
explanatory artwork, not a screenshot or measured memory/performance display.
All text has a fixed background; the assets work against light or dark pages.
No external fonts, scripts or asset services are required.

The existing Studio captures are actual interface images. The Chat image shows
an empty starting state. New art does not invent model answers, benchmark
numbers, download counts or testimonials.

## Visitor path

The README leads with the benefit, then the app, three setup steps, concrete
historical evidence and compatibility. [Start here](START_HERE.md) explains the
first conversation and common terms. [Installation](install.md) gives exact
package commands. [The engineering guide](ENGINEERING_GUIDE.md) retains the
former README's deeper commands, policies and historical measurements.

The downloads link points to the release list. Versioned release notes contain
the exact package links; the work log records publication status.
Windows GPU inference remains explicitly deferred/unqualified; production Qwen
admission stays OFF. Signed updater checks and Authenticode status are separate.

## Publishing this presentation

The user authorized merging the checked rebrand and publishing the update. The
repository's protected `main` follows the [PR merge policy](BRANCH_POLICY.md).
The README and new guides become the default presentation through PR integration.
Update the repository About description/topics to match the copy above. The PNG
is prepared for the social preview setting; creating the file does not mean
that setting has been changed.

Release tags, artifact upload and a published GitHub release remain separate
from the source push. Do not represent a branch push as a completed release.
