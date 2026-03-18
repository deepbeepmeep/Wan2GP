Hostile Design Review: `rosy-watching-hearth.md` vs current Headless-Wan2GP

Scope

This is a hostile, fresh-eyes review of [/Users/peteromalley/.claude/plans/rosy-watching-hearth.md](/Users/peteromalley/.claude/plans/rosy-watching-hearth.md) against the current Headless-Wan2GP codebase. The goal is to identify contradictions, missing plumbing, invalid assumptions, silent failure modes, and ambiguous implementation seams before work proceeds.

Findings

1. CRITICAL: `image_refs_strengths` will be silently dropped at the WGP boundary unless vendored signatures are updated first.
Current WGP sanitization uses `inspect.signature(wgp.generate_video)` and drops any key not present in the explicit signature: [source/models/wgp/orchestrator.py:913](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/orchestrator.py:913), [Wan2GP/wgp.py:5738](/Users/peteromalley/Documents/Headless-Wan2GP/Wan2GP/wgp.py:5738). The plan correctly notices this at lines 54-56 and 336-340 of the plan file, but it still stages `image_refs_strengths` wiring earlier through `StructureOutputs` and `_build_generation_params()` at plan lines 282-340. If an engineer implements the phase-5 plumbing before phase 7, per-anchor strengths will be dropped with only a debug log. This needs to be a hard phase gate, not an implied dependency.

2. CRITICAL: The plan’s alignment invariant breaks today on partial `image_refs_paths` load failure.
`prepare_svi_image_refs()` silently skips bad files and keeps going: [source/models/wgp/generators/preflight.py:20](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/generators/preflight.py:20). That means `image_refs_paths`, `frames_positions`, and any future `image_refs_strengths` can desynchronize if even one image fails to open. The plan treats the triple `image_refs[i] / frames_positions[i] / image_refs_strengths[i]` as a hard invariant at plan lines 27-31, but current preflight behavior violates that invariant silently. This should fail hard on any load error, or filter positions and strengths in lockstep using returned success indices.

3. CRITICAL: The proposed anchors-only overlap fix is placed behind a dead early return.
Current overlap logic returns immediately when there are no videos: [source/task_handlers/travel/orchestrator.py:104](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:104). The plan says to add anchor-aware overlap handling “after line 104” at plan lines 179-194. That code would never run for anchors-only hybrid payloads, because the function already returned `bool(guidance_video_url)`. The anchor check has to move before that return, or the return condition must become “no videos and no anchors”.

4. MAJOR: Hybrid control-video plumbing in `guide_builder` is incomplete if the engineer reuses `_create_ltx_control_guide_video()` “exactly as-is”.
The plan says to reuse `_create_ltx_control_guide_video()` exactly as-is at plan lines 268-270. That is not enough. The actual guide-building path only treats `is_ltx_control` as a direct-control case: [source/task_handlers/travel/guide_builder.py:248](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/guide_builder.py:248), [source/task_handlers/travel/guide_builder.py:285](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/guide_builder.py:285). A new `ltx_hybrid` kind with control present will otherwise fall through the generic guide path or skip guide generation entirely. On the downstream side, LTX is not guarded the way VACE is; only VACE hard-errors when `video_guide` is missing: [source/models/wgp/orchestrator.py:649](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/orchestrator.py:649). Inference: without explicitly generalizing those `is_ltx_control` branches, a literal implementation can silently degrade from `VGKFI` to anchors-only behavior.

5. CRITICAL: If `needs_ic_lora()` is extended to hybrid as written, the auto-injected IC-LoRA strength source is wrong.
The plan introduces `control_strength` and says legacy `strength` is unused for hybrid: plan lines 137-140 and 165-169. But auto-injected IC-LoRA currently always uses `travel_guidance_config.strength`: [source/task_handlers/travel/orchestrator.py:143](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:143), [source/task_handlers/travel/orchestrator.py:150](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:150). If an engineer implements the plan literally, pose/depth/canny hybrid control can end up auto-injecting the union control LoRA at `0.0` or at the wrong scalar. That silently disables part of the control path exactly where the plan expects hybrid to extend existing `ltx_control` behavior.

6. MAJOR: `to_segment_payload()` has no way to serialize remapped per-segment anchors today, and the plan is ambiguous about who owns that remapping.
Current `TravelGuidanceConfig.to_segment_payload()` only knows about `kind`, `videos`, `strength`, `mode`, and existing legacy fields: [source/core/params/travel_guidance.py:216](/Users/peteromalley/Documents/Headless-Wan2GP/source/core/params/travel_guidance.py:216). The orchestrator currently passes one shared config into `_build_segment_travel_guidance_payload()` and then directly calls `config.to_segment_payload(frame_offset=...)`: [source/task_handlers/travel/orchestrator.py:123](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:123), [source/task_handlers/travel/orchestrator.py:1921](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:1921). The plan says segment-local anchor remapping happens in phase 3a, but it never pins down whether the orchestrator creates per-segment `TravelGuidanceConfig` objects, passes remapped anchors into `to_segment_payload()`, or bypasses `to_segment_payload()` for anchor injection. That is a real implementation gap, not a minor detail.

7. MAJOR: The plan has an internal contract contradiction on duplicate positions and validation scope.
At plan line 23, it says validation must reject duplicate positions at both global and segment-local levels. At plan lines 68-69, it says global duplicate positions are fine and only local duplicates should be rejected after remapping. Those are mutually exclusive rules. There is a second contradiction at plan lines 148-151 versus 160-161: it says one validation rule can be reused globally and locally, but also says anchor count is enforced per emitted segment. Those are not the same validation context.

8. MAJOR: The plan also contradicts itself on where the alignment invariant is enforced.
Plan lines 48-49 say filtering operates on `AnchorEntry` objects and conversion to parallel lists happens exactly once at `process_segment()` output. But plan line 266 says `_filter_anchors_for_segment()` returns `(local_image_urls, local_positions, local_strengths)` as a tuple of parallel lists. Those are different designs. An engineer can implement either and still think they are following the plan.

9. MAJOR: `validate()` and `has_guidance` are both load-bearing blockers for anchors-only hybrid.
Current validation rejects every non-`none` travel-guidance payload without videos: [source/core/params/travel_guidance.py:262](/Users/peteromalley/Documents/Headless-Wan2GP/source/core/params/travel_guidance.py:262). Current `has_guidance` only counts videos or `_guidance_video_url`: [source/core/params/travel_guidance.py:296](/Users/peteromalley/Documents/Headless-Wan2GP/source/core/params/travel_guidance.py:296). The plan does call for changing both at lines 148-151 and 72-74, but these are not “nice-to-have” updates. Without them, anchors-only hybrid cannot even be parsed, and even a parsed object would be treated as guidance-free by `_process_structure_guidance()`: [source/task_handlers/tasks/task_registry.py:547](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/tasks/task_registry.py:547), [source/task_handlers/tasks/task_registry.py:558](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/tasks/task_registry.py:558).

10. MAJOR: `_process_structure_guidance()` and `TravelSegmentProcessor.process_segment()` currently drop every new hybrid field unless both are extended together.
`StructureOutputs` only has four fields today: [source/task_handlers/tasks/task_registry.py:94](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/tasks/task_registry.py:94). `_process_structure_guidance()` only extracts `video_guide`, `video_mask`, `video_prompt_type`, and `structure_type` from the processor result: [source/task_handlers/tasks/task_registry.py:592](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/tasks/task_registry.py:592). `TravelSegmentProcessor.process_segment()` currently only returns those four keys: [source/task_handlers/travel/segment_processor.py:261](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/segment_processor.py:261). The plan mentions extending all three places, but this is the most obvious silent-drop seam in the actual codebase.

11. MAJOR: The plan over-focuses on `task_conversion.py`, but the live travel-segment path needs separate verification.
Travel segments do not go through `db_task_to_generation_task()` on the main queue path. They build `generation_params` directly and submit `GenerationTask(parameters=generation_params)` from `_handle_travel_segment_via_queue_impl()`: [source/task_handlers/tasks/task_registry.py:1027](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/tasks/task_registry.py:1027), [source/task_handlers/tasks/task_registry.py:1077](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/tasks/task_registry.py:1077). So phase 5d is incomplete and slightly mis-aimed. It is incomplete because `image_refs_paths` is also missing from the whitelist: [source/task_handlers/tasks/task_conversion.py:38](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/tasks/task_conversion.py:38). It is also not sufficient on its own, because the critical travel-segment transport path instead relies on `TaskConfig.extra_params` passthrough: [source/core/params/task.py:73](/Users/peteromalley/Documents/Headless-Wan2GP/source/core/params/task.py:73), [source/core/params/task.py:140](/Users/peteromalley/Documents/Headless-Wan2GP/source/core/params/task.py:140).

12. MAJOR: `wgp.py` forwarding of `image_refs_strengths` is still underspecified.
The plan says “pass through from kwargs to the LTX handler” at lines 56 and 336-340 of the plan, but WGP does not forward arbitrary kwargs at the actual model call site. It makes an explicit `wan_model.generate(...)` call: [Wan2GP/wgp.py:6761](/Users/peteromalley/Documents/Headless-Wan2GP/Wan2GP/wgp.py:6761). `ltx2.generate()` can accept extra kwargs because it has `**kwargs`, but only if WGP explicitly passes `image_refs_strengths` into that call. The exact call site exists; the plan should name it directly.

13. MODERATE: `create_video_prompt_type()` has no hybrid branch, and its LTX fallback is silently wrong.
Right now only `is_ltx_control` gets a travel-guidance override: [source/task_handlers/travel/segment_processor.py:120](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/segment_processor.py:120). If a new hybrid kind falls through, the LTX fallback path emits `"S"` when no mask is present, which is start-image mode, not hybrid anchor/control mode. The plan calls for a hybrid branch at lines 236-247, which is correct, but this is a silent correctness bug during implementation, not an obvious crash.

14. MODERATE: The pre-implementation VGKFI gate is necessary, but it is still too weak if it only exercises raw WGP.
The plan’s first gate at lines 80-102 is directionally right, but it does not test the actual travel path that must produce `travel_guidance` payloads, materialize `video_guide`, set `video_prompt_type`, propagate fields through `StructureOutputs`, survive queue conversion, and then hit WGP. A raw WGP semantic test can pass while the real travel implementation still fails in `guide_builder`, `task_registry`, or direct queue submission.

15. MODERATE: `image_refs_paths` conversion is sound, but adding it to the WGP signature later could create a double-source ambiguity.
Today `prepare_svi_image_refs()` converts `image_refs_paths` into `image_refs` before WGP param resolution: [source/models/wgp/orchestrator.py:575](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/orchestrator.py:575), [source/models/wgp/generators/preflight.py:13](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/generators/preflight.py:13). That is a good reuse path. But the plan should explicitly keep `image_refs_paths` out of `wgp.generate_video()`’s signature. If someone later adds it there “for completeness”, the system can end up with both `image_refs` and `image_refs_paths` in play, which is fragile.

16. MODERATE: Anchor remapping is ambiguous because the orchestrator uses two offset systems, but the plan speaks as if there is only one.
Travel-guidance overlap uses stitched offsets: [source/task_handlers/travel/orchestrator.py:107](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:107). Segment payloads use `segment_frame_offset`, which can come from stitched offsets or flow offsets depending on `use_stitched_offsets`: [source/task_handlers/travel/orchestrator.py:1888](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:1888). The plan’s anchor remapping language at lines 204-212 says “compute segment_frame_start” but never pins down which timeline anchors live on. If remapping uses the wrong offset basis, overlap gating and payload-local positions can disagree.

17. MINOR: Audio slicing ownership is still fuzzy.
The plan wants external audio downloaded once at orchestrator level, then sliced per segment, with slices living in `segment_processing_dir`: plan lines 216-220 and 64-65. But `segment_processing_dir` is a segment-worker concern, not a natural orchestrator concern; the segment processor only learns it inside [source/task_handlers/tasks/task_registry.py:577](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/tasks/task_registry.py:577). The design can work, but the lifecycle boundary is not fully specified.

18. MINOR: `AnchorEntry.from_dict()` / `to_dict()` need explicit global-vs-local semantics.
The plan stubs these methods at lines 119-131 but does not define whether `frame_position` in the serialized form is global or segment-local. That matters because the same dataclass is supposed to represent both global configs and emitted per-segment payloads.

Open Questions

1. Who owns segment-local hybrid construction?
The current orchestrator only has one shared `TravelGuidanceConfig` object and calls `config.to_segment_payload(frame_offset=...)`: [source/task_handlers/travel/orchestrator.py:123](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:123). The plan needs to say whether per-segment configs are cloned, whether `to_segment_payload()` accepts segment-local anchors, or whether the orchestrator injects hybrid fields outside `to_segment_payload()`.

2. What is the exact `wgp.py` call site change for `image_refs_strengths`?
The forwarding seam is not “somewhere in WGP”; it is the explicit `wan_model.generate(...)` call near [Wan2GP/wgp.py:6761](/Users/peteromalley/Documents/Headless-Wan2GP/Wan2GP/wgp.py:6761). The plan should name that site directly.

3. Should preflight fail hard on any bad anchor image?
Given the stated hard alignment invariant, permissive partial loads in [source/models/wgp/generators/preflight.py:20](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/generators/preflight.py:20) are probably the wrong behavior.

4. Which scalar drives IC-LoRA strength for hybrid: `control_strength`, legacy `strength`, or something else?
Right now the auto-injection path only knows about `.strength`: [source/task_handlers/travel/orchestrator.py:146](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:146). The plan needs to resolve this explicitly.

5. Which timeline are anchor frame positions expressed on?
The plan talks about “journey-global” positions, but the orchestrator has stitched and flow offsets. The contract should name one timeline and require remapping from that timeline only.

What Looks Sound

- Reusing native Wan2GP/WGP mechanisms is the right architecture. The plan is correctly trying to compose `image_refs`, `frames_positions`, `video_prompt_type`, and optional audio rather than invent a parallel pipeline.
- Reusing `prepare_svi_image_refs()` for `image_refs_paths -> image_refs` is correct in principle: [source/models/wgp/orchestrator.py:575](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/orchestrator.py:575), [source/models/wgp/generators/preflight.py:13](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/generators/preflight.py:13).
- The per-anchor-strength change in `ltx2.py` is genuinely small and backward-compatible if the new list is optional: [Wan2GP/models/ltx2/ltx2.py:965](/Users/peteromalley/Documents/Headless-Wan2GP/Wan2GP/models/ltx2/ltx2.py:965).
- The plan is right that mixed prompt strings like `VGKFI` must be proven semantically, not just syntactically.
- The audio-system split is real. Generation-time audio and post-stitch mux audio are separate concerns in the current codebase.
- The requirement that `audio_prompt_type="K"` only makes sense with control video matches WGP’s rules: [Wan2GP/wgp.py:834](/Users/peteromalley/Documents/Headless-Wan2GP/Wan2GP/wgp.py:834), [Wan2GP/wgp.py:6102](/Users/peteromalley/Documents/Headless-Wan2GP/Wan2GP/wgp.py:6102).

Direct Answers

If an engineer implemented this plan literally, where would it most likely break first?

The first likely break is still the anchors-only overlap path. `_segment_has_travel_guidance_overlap()` returns early when there are no videos: [source/task_handlers/travel/orchestrator.py:104](/Users/peteromalley/Documents/Headless-Wan2GP/source/task_handlers/travel/orchestrator.py:104). Because the plan places the new anchor logic after that return, a literal implementation would still emit `{"kind": "none"}` for anchors-only hybrid segments. That is the cleanest proven break from the current code.

What is the most likely silent bug?

The most likely silent bug is the alignment break on partial anchor-image load failure. A single unreadable file in `image_refs_paths` causes `image_refs` to shrink while `frames_positions` and `image_refs_strengths` stay untouched: [source/models/wgp/generators/preflight.py:20](/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/generators/preflight.py:20). The feature will appear to “work”, but anchors and strengths will be applied to the wrong positions.

What single assumption in the plan is riskiest?

The riskiest assumption is still that mixed `video_prompt_type` strings like `VGKFI` and `PVGKFI` behave correctly all the way through WGP preprocessing and into LTX conditioning. The plan is right to gate on this. But the gate should be extended to the real travel path, not just a raw WGP unit test, because the actual failure surface includes `guide_builder`, `task_registry`, and queue transport as well as WGP itself.
