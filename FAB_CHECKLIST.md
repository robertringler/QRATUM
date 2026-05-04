# QRATUM — FAB UPLOAD CHECKLIST

---

## 1. PRE-PACKAGE VERIFICATION

- [ ] QRATUM.uplugin exists in plugin root
- [ ] FileVersion = 3
- [ ] VersionName = "1.0.0"
- [ ] All module names match Source/ folder exactly (QRATUMCore, QRATUMPerception, QRATUMMemory, QRATUMActions)
- [ ] No circular dependencies in Build.cs files
- [ ] QRATUMCore has zero dependencies on sibling modules
- [ ] MemoryStructs.h located at Source/QRATUMCore/Public/
- [ ] Plugin compiles in Development Editor configuration
- [ ] Plugin compiles in Shipping configuration
- [ ] No deprecated UE4 APIs present
- [ ] No hardcoded asset paths in C++  ⚠️ COMMON REJECTION CAUSE

---

## 2. IN-EDITOR VALIDATION (UE5.7)

- [ ] Plugin loads in clean project with no Output Log errors
- [ ] QRATUMBrainComponent can be added to Pawn BP
- [ ] QRATUMPerceptionComponent fires OnPerceivedActor correctly
- [ ] QRATUMMemoryComponent runs decay without leaks (10-min session test)
- [ ] QRATUMActionExecutor — all 5 behaviors execute without crash:
  - [ ] Patrol
  - [ ] Chase
  - [ ] Flee
  - [ ] Combat (delegate fires, no animation dependency)
  - [ ] Search (falls back to Patrol after SearchTimeout)
- [ ] Demo map opens without missing assets  ⚠️ COMMON REJECTION CAUSE
- [ ] Full Perception → Memory → Decision → Action loop runs in demo map
- [ ] Debug overlay mounts via BindToOwner()
- [ ] QRATUM.Debug 1 shows widget
- [ ] QRATUM.Debug 0 removes widget cleanly
- [ ] No null pointer crashes during PIE session

---

## 3. ASSET PREPARATION

- [ ] Thumbnail: 894×894px square  ⚠️ REQUIRED DIMENSION
- [ ] Thumbnail readable at 200×200px
- [ ] No code screenshots in any image
- [ ] Minimum 3 gallery images:
  - [ ] Image 1: system flow diagram (Perception → Memory → Decision → Action)
  - [ ] Image 2: in-engine NPC behavior screenshot
  - [ ] Image 3: debug overlay showing live cognitive state

### Video requirements (Fab)
- [ ] Format: MP4, H.264 codec  ⚠️ REQUIRED FORMAT
- [ ] Resolution: 1920×1080 minimum
- [ ] Frame rate: 30 or 60 FPS
- [ ] Duration: 30–90 seconds
- [ ] No external watermarks
- [ ] First 5 seconds show core value (NPC reacting to player)
- [ ] Readable without sound (no audio dependency for comprehension)

---

## 4. LISTING COPY VALIDATION

- [ ] Title contains: Unreal Engine + AI + "Behavior Tree"
- [ ] Description lists ONLY features present in shipped package
- [ ] No roadmap features listed as current  ⚠️ REJECTION / DELISTING RISK
- [ ] Tier 1 ($49) is the only active purchasable SKU
- [ ] V1.1 and V2.0 explicitly marked as roadmap, not included
- [ ] Tags include: AI, NPC, Behavior Tree replacement, Unreal Engine 5 AI
- [ ] No claims about LLM, ML, or neural systems
- [ ] Supported engine version explicitly stated: UE5.7+
- [ ] Installation steps included in description

---

## 5. SUBMISSION

- [ ] Plugin packaged as .zip with QRATUM.uplugin at root
- [ ] Source/ directory complete and included
- [ ] Content/ directory complete (DemoMap included)
- [ ] No missing module binaries
- [ ] Upload completes without Fab validation errors
- [ ] Preview page renders correctly before publish
- [ ] All media assets load in preview

### Common rejection flags
- ⛔ Missing module in Build.cs dependency chain
- ⛔ Plugin fails to load in clean UE5.7 project
- ⛔ Hardcoded asset paths in C++ source
- ⛔ Unreal version mismatch in .uplugin
- ⛔ Referenced Content/ assets missing from package
- ⛔ Features described but not implemented

### If rejected
- [ ] Check Output Log for missing dependency errors
- [ ] Verify .uplugin module names match compiled binaries exactly
- [ ] Rebuild in Development Editor before resubmission
- [ ] Confirm all Blueprint-referenced assets exist in Content/

---

## 6. LAUNCH DAY

- [ ] Publish Tuesday–Thursday, 9:00–11:00 AM EST
- [ ] Primary thumbnail = Variant 2 ("NO MORE BEHAVIOR TREES")
- [ ] Video autoplay enabled in Fab preview
- [ ] Post in r/unrealengine (technical framing, not sales)
- [ ] Post in Unreal Slackers Discord
- [ ] Post in indie dev communities
- [ ] First post is a question / observation, not a sales link
- [ ] No pricing changes within first 72 hours
- [ ] Ask first 3–10 buyers explicitly for a review
- [ ] Monitor CTR and conversion rate hourly on day 1

---

If every box above is checked: QRATUM is a shipped marketplace product.
