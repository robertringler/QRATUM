# QRATUM Core AI System — v1.0.0

Autonomous NPC runtime for Unreal Engine 5.7.
Replaces Behavior Trees with a deterministic Perception → Memory → Decision → Action pipeline.

---

## Requirements

- Unreal Engine 5.7+
- AIController-based Pawn setup
- Navigation System enabled in project

---

## Installation

1. Copy the QRATUM/ folder into your project's Plugins/ directory
2. Regenerate project files
3. Enable the plugin in Edit → Plugins → AI → QRATUM Core AI System
4. Assign your WBP_QRATUMDebug asset in Project Settings → QRATUM (optional, required for debug overlay)

---

## Quick Start

1. Add the following components to your AI Pawn Blueprint:
   - QRATUMBrainComponent
   - QRATUMPerceptionComponent
   - QRATUMMemoryComponent
   - QRATUMActionExecutor

2. In the Pawn Blueprint, bind delegates:
   - OnPerceivedActor → AddIntent (map Sighted to Chase, Heard to Investigate)
   - OnDecisionMade → ExecuteChase / ExecuteFlee / ExecutePatrol / ExecuteSearch / ExecuteCombat

3. Set PerceptionObjectTypes on QRATUMPerceptionComponent to include Pawn

4. Call ProcessSight() on a repeating timer (recommended: 0.2-0.5s interval)

---

## Module Overview

| Module             | Responsibility                          |
|--------------------|----------------------------------------|
| QRATUMCore         | Brain, intent evaluation, health interface |
| QRATUMPerception   | Sight cone, hearing events             |
| QRATUMMemory       | Salience-based memory with decay/prune |
| QRATUMActions      | Movement, combat signal, flee, search  |

---

## Dependency Graph

QRATUMPerception  ─┐
QRATUMMemory       ├──→ QRATUMCore (no reverse dependency)
QRATUMActions     ─┘

---

## Debug Overlay

Enable in PIE with console command:
  QRATUM.Debug 1    (show)
  QRATUM.Debug 0    (hide)

Requires DebugWidgetClass assigned in Project Settings → QRATUM.

---

## UMG Layout Spec (WBP_QRATUMDebug)

Create a UserWidget Blueprint based on UQRATUMDebugWidget.
Widget hierarchy:

  Canvas Panel (root)
  │
  ├── Border (semi-transparent black, opacity 0.5)
  │   └── Vertical Box
  │       │
  │       ├── TextBlock — name: "CurrentIntentText"
  │       │   Font: Bold, 18px
  │       │   Label: "Intent: "
  │       │
  │       ├── TextBlock — name: "LastPerceptionText"
  │       │   Font: Regular, 14px
  │       │   Label: "Perception: "
  │       │
  │       └── Vertical Box — name: "MemoryList"
  │           (populated at runtime by RefreshMemoryUI)
  │           Font: Monospace, 12px
  │
  └── TextBlock (top-left anchor)
      Content: "QRATUM DEBUG"
      Font: Bold, 11px, color white

Widget names MUST match exactly: CurrentIntentText, LastPerceptionText, MemoryList.
Call BindToOwner(YourNPCActor) after AddToViewport to activate.

---

## Optional: Health Interface

To supply real health values to the Brain, implement IQRATUMHealthProvider on your Pawn:

  virtual float GetNormalizedHealth() const override
  {
      return CurrentHealth / MaxHealth;
  }

If not implemented, Brain defaults to 1.0f (full health assumed).

---

## V1 Behaviors

- Patrol   — cycles through waypoint array
- Chase    — AIMoveTo target actor location
- Flee     — AIMoveTo inverted vector from threat
- Combat   — fires OnCombatTriggered delegate (bind your damage logic)
- Search   — AIMoveTo last known location, falls back to Patrol on timeout

---

## Roadmap (not included in V1.0)

V1.1 — Expanded intent library (Investigate, Guard, Search zones)
V2.0 — Multi-agent coordination (shared memory propagation, squad roles)

---

## Fab Listing

Title: QRATUM Core AI – Behavior Tree Replacement for Autonomous NPCs (UE5.7)
Price: $49
Category: AI / Gameplay Systems
Tags: Behavior Tree replacement, NPC AI system, Unreal Engine 5 AI, autonomous NPC,
      AI controller system, perception system UE5, decision system Unreal, game AI framework
