# Subsystem contract: `reality_interface`
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

**Path**: `qagents/{reality_interface.py, reality_interface_v2.py, llm_backends.py, ciir_ric_bridge.py, prompts/}`

## Files
- reality_interface.py (deterministic 4-key output)
- reality_interface_v2.py (k-step rollout + anti-deadlock + entropy injection)
- llm_backends.py (DeterministicLLM + OpenAI/Anthropic/Gemini/Local with fallback)
- ciir_ric_bridge.py (snapshot→world_state, action→learning_rate; opt-in v2)

## Public types / API surface
- Strict 4-key output schema (intent_interpretation, selected_action, predicted_outcome, fallback_plan)
- action types: control|adjust|hold|abort
- RICv2Controller, RICv2History (trajectory_summary, stability_projection, risk_profile)

## Invariants
- v1 emits exactly 4 keys
- v2 strictly extends v1 (4 base keys preserved)
