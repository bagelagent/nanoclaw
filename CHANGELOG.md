# Changelog

All notable changes to NanoClaw will be documented in this file.

## Unreleased

- **feat:** Add ACE-Step 1.5 music generation via ComfyUI (`comfyui_music` MCP tool). Supports tags, lyrics, BPM, key/scale, time signature, and duration. Uses local GPU via the same IPC pattern as `comfyui_generate`.
- **docs:** Update SPEC.md MCP tools table with all media/messaging tools. Fix stale credential filtering docs.

## [1.2.0](https://github.com/qwibitai/nanoclaw/compare/v1.1.6...v1.2.0)

[BREAKING] WhatsApp removed from core, now a skill. Run `/add-whatsapp` to re-add (existing auth/groups preserved).
