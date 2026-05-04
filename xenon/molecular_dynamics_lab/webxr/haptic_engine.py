"""Haptic feedback engine for molecular interactions.

Provides haptic feedback simulation for VR molecular visualization,
translating molecular forces and interactions into vibration patterns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HapticPattern(Enum):
    """Predefined haptic feedback patterns."""

    ATOM_SELECT = "atom_select"
    ATOM_HOVER = "atom_hover"
    BOND_FORM = "bond_form"
    BOND_BREAK = "bond_break"
    COLLISION = "collision"
    REPULSION = "repulsion"
    ATTRACTION = "attraction"
    SURFACE_CONTACT = "surface_contact"
    ENERGY_MINIMUM = "energy_minimum"
    CONSTRAINT = "constraint"
    ERROR = "error"
    SUCCESS = "success"


class HapticChannel(Enum):
    """Haptic output channels."""

    LEFT = "left"
    LEFT_HAND = "left"
    RIGHT = "right"
    RIGHT_HAND = "right"
    BOTH = "both"


@dataclass
class HapticConfig:
    """Configuration for haptic feedback."""

    enabled: bool = True
    intensity_scale: float = 1.0  # Global intensity multiplier
    force_feedback_scale: float = 0.1  # Scale molecular forces to haptics
    min_intensity: float = 0.0
    max_intensity: float = 1.0
    default_duration: float = 100.0  # milliseconds
    force_cutoff: float = 10.0  # kcal/mol/Angstrom
    enable_continuous_feedback: bool = True
    continuous_update_rate: float = 60.0  # Hz

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "enabled": self.enabled,
            "intensityScale": self.intensity_scale,
            "forceFeedbackScale": self.force_feedback_scale,
            "minIntensity": self.min_intensity,
            "maxIntensity": self.max_intensity,
            "defaultDuration": self.default_duration,
            "forceCutoff": self.force_cutoff,
            "enableContinuousFeedback": self.enable_continuous_feedback,
            "continuousUpdateRate": self.continuous_update_rate,
        }


@dataclass
class HapticPulse:
    """Single haptic pulse specification."""

    intensity: float  # 0.0 to 1.0
    duration: float  # milliseconds
    frequency: float = 0.0  # Hz (0 = default actuator frequency)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "intensity": self.intensity,
            "duration": self.duration,
            "frequency": self.frequency,
        }


@dataclass
class HapticWaveform:
    """Custom haptic waveform."""

    pulses: list[HapticPulse] = field(default_factory=list)
    loop: bool = False
    name: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pulses": [p.to_dict() for p in self.pulses],
            "loop": self.loop,
            "name": self.name,
        }

    @classmethod
    def from_pattern(cls, pattern: HapticPattern) -> HapticWaveform:
        """Create waveform from predefined pattern.

        Args:
            pattern: HapticPattern enum value

        Returns:
            HapticWaveform for the pattern
        """
        waveforms = {
            HapticPattern.ATOM_SELECT: cls(
                name="atom_select",
                pulses=[HapticPulse(0.8, 50), HapticPulse(0.0, 30), HapticPulse(0.4, 30)],
            ),
            HapticPattern.ATOM_HOVER: cls(
                name="atom_hover",
                pulses=[HapticPulse(0.2, 20)],
            ),
            HapticPattern.BOND_FORM: cls(
                name="bond_form",
                pulses=[
                    HapticPulse(0.3, 30),
                    HapticPulse(0.6, 50),
                    HapticPulse(0.9, 80),
                    HapticPulse(0.5, 40),
                ],
            ),
            HapticPattern.BOND_BREAK: cls(
                name="bond_break",
                pulses=[
                    HapticPulse(0.9, 60),
                    HapticPulse(0.0, 20),
                    HapticPulse(0.5, 40),
                    HapticPulse(0.0, 20),
                    HapticPulse(0.2, 30),
                ],
            ),
            HapticPattern.COLLISION: cls(
                name="collision",
                pulses=[HapticPulse(1.0, 100, 200)],
            ),
            HapticPattern.REPULSION: cls(
                name="repulsion",
                pulses=[
                    HapticPulse(0.6, 20, 150),
                    HapticPulse(0.4, 20, 150),
                    HapticPulse(0.2, 20, 150),
                ],
                loop=True,
            ),
            HapticPattern.ATTRACTION: cls(
                name="attraction",
                pulses=[
                    HapticPulse(0.2, 50, 50),
                    HapticPulse(0.4, 50, 50),
                    HapticPulse(0.6, 50, 50),
                ],
                loop=True,
            ),
            HapticPattern.SURFACE_CONTACT: cls(
                name="surface_contact",
                pulses=[HapticPulse(0.3, 30), HapticPulse(0.5, 50)],
            ),
            HapticPattern.ENERGY_MINIMUM: cls(
                name="energy_minimum",
                pulses=[
                    HapticPulse(0.9, 100),
                    HapticPulse(0.7, 80),
                    HapticPulse(0.5, 60),
                    HapticPulse(0.3, 40),
                    HapticPulse(0.1, 20),
                ],
            ),
            HapticPattern.CONSTRAINT: cls(
                name="constraint",
                pulses=[HapticPulse(0.5, 100, 100)],
                loop=True,
            ),
            HapticPattern.ERROR: cls(
                name="error",
                pulses=[
                    HapticPulse(1.0, 100),
                    HapticPulse(0.0, 100),
                    HapticPulse(1.0, 100),
                ],
            ),
            HapticPattern.SUCCESS: cls(
                name="success",
                pulses=[
                    HapticPulse(0.3, 50),
                    HapticPulse(0.6, 75),
                    HapticPulse(0.9, 100),
                ],
            ),
        }

        return waveforms.get(pattern, cls(name=pattern.value, pulses=[HapticPulse(0.5, 50)]))


@dataclass
class HapticFeedback:
    """Haptic feedback event."""

    channel: HapticChannel
    waveform: HapticWaveform
    timestamp: float = 0.0
    source: str = ""  # What triggered this feedback

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "channel": self.channel.value,
            "waveform": self.waveform.to_dict(),
            "timestamp": self.timestamp,
            "source": self.source,
        }


class HapticEngine:
    """Haptic feedback engine for molecular interactions.

    Translates molecular forces and events into haptic feedback
    for VR controllers and haptic devices.
    """

    def __init__(self, config: Optional[HapticConfig] = None):
        """Initialize haptic engine.

        Args:
            config: Haptic configuration
        """
        self.config = config or HapticConfig()
        self._active_waveforms: dict[str, HapticWaveform] = {}
        self._force_buffer: dict[str, np.ndarray] = {
            "left": np.zeros(3),
            "right": np.zeros(3),
        }
        self._callbacks: list[Callable[[HapticFeedback], None]] = []

    def trigger_pattern(
        self,
        pattern: HapticPattern,
        channel: HapticChannel = HapticChannel.BOTH,
        intensity_modifier: float = 1.0,
    ) -> HapticFeedback:
        """Trigger a predefined haptic pattern.

        Args:
            pattern: Haptic pattern to trigger
            channel: Output channel(s)
            intensity_modifier: Intensity multiplier

        Returns:
            HapticFeedback event
        """
        if not self.config.enabled:
            return HapticFeedback(
                channel=channel,
                waveform=HapticWaveform(),
                source=pattern.value,
            )

        waveform = HapticWaveform.from_pattern(pattern)

        # Apply intensity modifier
        for pulse in waveform.pulses:
            pulse.intensity = self._clamp_intensity(
                pulse.intensity * intensity_modifier * self.config.intensity_scale
            )

        feedback = HapticFeedback(
            channel=channel,
            waveform=waveform,
            source=pattern.value,
        )

        self._dispatch_feedback(feedback)
        return feedback

    def trigger_pulse(
        self,
        intensity: float,
        duration: float,
        channel: HapticChannel = HapticChannel.BOTH,
        frequency: float = 0.0,
    ) -> HapticFeedback:
        """Trigger a single haptic pulse.

        Args:
            intensity: Pulse intensity (0-1)
            duration: Duration in milliseconds
            channel: Output channel(s)
            frequency: Vibration frequency in Hz

        Returns:
            HapticFeedback event
        """
        if not self.config.enabled:
            return HapticFeedback(
                channel=channel,
                waveform=HapticWaveform(),
                source="pulse",
            )

        pulse = HapticPulse(
            intensity=self._clamp_intensity(intensity * self.config.intensity_scale),
            duration=duration,
            frequency=frequency,
        )

        waveform = HapticWaveform(pulses=[pulse], name="custom_pulse")

        feedback = HapticFeedback(
            channel=channel,
            waveform=waveform,
            source="pulse",
        )

        self._dispatch_feedback(feedback)
        return feedback

    def trigger_custom_waveform(
        self,
        waveform: HapticWaveform,
        channel: HapticChannel = HapticChannel.BOTH,
    ) -> HapticFeedback:
        """Trigger a custom haptic waveform.

        Args:
            waveform: Custom waveform
            channel: Output channel(s)

        Returns:
            HapticFeedback event
        """
        if not self.config.enabled:
            return HapticFeedback(
                channel=channel,
                waveform=HapticWaveform(),
                source="custom",
            )

        # Apply intensity scaling
        scaled_waveform = HapticWaveform(
            pulses=[
                HapticPulse(
                    intensity=self._clamp_intensity(p.intensity * self.config.intensity_scale),
                    duration=p.duration,
                    frequency=p.frequency,
                )
                for p in waveform.pulses
            ],
            loop=waveform.loop,
            name=waveform.name,
        )

        feedback = HapticFeedback(
            channel=channel,
            waveform=scaled_waveform,
            source="custom",
        )

        self._dispatch_feedback(feedback)
        return feedback

    def update_force_feedback(
        self,
        force: np.ndarray,
        channel: HapticChannel = HapticChannel.RIGHT,
    ) -> Optional[HapticFeedback]:
        """Update continuous force feedback.

        Converts molecular force to haptic intensity.

        Args:
            force: Force vector (kcal/mol/Angstrom)
            channel: Output channel

        Returns:
            HapticFeedback if force is significant
        """
        if not self.config.enabled or not self.config.enable_continuous_feedback:
            return None

        # Store force in buffer
        hand = channel.value if channel != HapticChannel.BOTH else "right"
        self._force_buffer[hand] = force

        # Calculate force magnitude
        force_mag = np.linalg.norm(force)

        if force_mag < 0.01:  # Negligible force
            return None

        # Map force to intensity
        intensity = min(force_mag / self.config.force_cutoff, 1.0)
        intensity *= self.config.force_feedback_scale

        if intensity < self.config.min_intensity:
            return None

        # Calculate frequency based on force direction
        # Higher frequency for repulsion, lower for attraction
        frequency = 50 + 150 * intensity  # 50-200 Hz range

        # Create feedback pulse
        pulse = HapticPulse(
            intensity=self._clamp_intensity(intensity),
            duration=1000 / self.config.continuous_update_rate,
            frequency=frequency,
        )

        waveform = HapticWaveform(pulses=[pulse], name="force_feedback")

        feedback = HapticFeedback(
            channel=channel,
            waveform=waveform,
            source="force",
        )

        self._dispatch_feedback(feedback)
        return feedback

    def feedback_from_molecular_event(
        self,
        event_type: str,
        event_data: dict,
        hand: str = "right",
    ) -> Optional[HapticFeedback]:
        """Generate haptic feedback from molecular event.

        Args:
            event_type: Type of molecular event
            event_data: Event data
            hand: Which hand triggered the event

        Returns:
            HapticFeedback if applicable
        """
        channel = HapticChannel.LEFT_HAND if hand == "left" else HapticChannel.RIGHT_HAND

        event_patterns = {
            "atom_selected": HapticPattern.ATOM_SELECT,
            "atom_hovered": HapticPattern.ATOM_HOVER,
            "bond_formed": HapticPattern.BOND_FORM,
            "bond_broken": HapticPattern.BOND_BREAK,
            "collision_detected": HapticPattern.COLLISION,
            "surface_contact": HapticPattern.SURFACE_CONTACT,
            "energy_minimum_found": HapticPattern.ENERGY_MINIMUM,
            "constraint_violated": HapticPattern.CONSTRAINT,
            "docking_success": HapticPattern.SUCCESS,
            "docking_failed": HapticPattern.ERROR,
        }

        pattern = event_patterns.get(event_type)
        if pattern:
            # Adjust intensity based on event data
            intensity_modifier = event_data.get("intensity", 1.0)
            return self.trigger_pattern(pattern, channel, intensity_modifier)

        return None

    def feedback_from_interaction_energy(
        self,
        energy: float,
        hand: str = "right",
        energy_range: tuple[float, float] = (-50.0, 50.0),
    ) -> Optional[HapticFeedback]:
        """Generate feedback based on interaction energy.

        Args:
            energy: Interaction energy in kcal/mol
            hand: Which hand
            energy_range: Expected energy range (min, max)

        Returns:
            HapticFeedback based on energy
        """
        channel = HapticChannel.LEFT_HAND if hand == "left" else HapticChannel.RIGHT_HAND

        # Normalize energy to -1 to 1 range
        e_min, e_max = energy_range
        normalized = 2 * (energy - e_min) / (e_max - e_min) - 1
        normalized = max(-1, min(1, normalized))

        # Negative energy (favorable) -> gentle pulsing
        # Positive energy (unfavorable) -> stronger, higher frequency
        if normalized < 0:
            # Favorable interaction - attraction pattern
            intensity = abs(normalized) * 0.5
            frequency = 50 + 50 * abs(normalized)
            pattern = HapticPattern.ATTRACTION
        else:
            # Unfavorable interaction - repulsion pattern
            intensity = normalized * 0.8
            frequency = 100 + 150 * normalized
            pattern = HapticPattern.REPULSION

        waveform = HapticWaveform.from_pattern(pattern)
        for pulse in waveform.pulses:
            pulse.intensity = self._clamp_intensity(intensity)
            pulse.frequency = frequency

        feedback = HapticFeedback(
            channel=channel,
            waveform=waveform,
            source=f"energy_{energy:.1f}",
        )

        self._dispatch_feedback(feedback)
        return feedback

    def stop_continuous_feedback(self, channel: HapticChannel = HapticChannel.BOTH) -> None:
        """Stop continuous feedback on specified channel(s).

        Args:
            channel: Channel(s) to stop
        """
        hands = ["left", "right"] if channel == HapticChannel.BOTH else [channel.value]

        for hand in hands:
            if hand in self._active_waveforms:
                del self._active_waveforms[hand]
            self._force_buffer[hand] = np.zeros(3)

    def add_callback(self, callback: Callable[[HapticFeedback], None]) -> None:
        """Add callback for haptic feedback events.

        Args:
            callback: Function called on feedback events
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[HapticFeedback], None]) -> None:
        """Remove callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _clamp_intensity(self, intensity: float) -> float:
        """Clamp intensity to valid range.

        Args:
            intensity: Raw intensity value

        Returns:
            Clamped intensity
        """
        return max(self.config.min_intensity, min(self.config.max_intensity, intensity))

    def _dispatch_feedback(self, feedback: HapticFeedback) -> None:
        """Dispatch feedback to callbacks.

        Args:
            feedback: Feedback event to dispatch
        """
        for callback in self._callbacks:
            try:
                callback(feedback)
            except Exception as e:
                logger.error(f"Haptic callback error: {e}")

    def generate_haptic_js(self) -> str:
        """Generate JavaScript for WebXR haptic feedback.

        Returns:
            JavaScript code for haptics
        """
        config_json = json.dumps(self.config.to_dict())

        return f"""
// Haptic Feedback Engine Configuration
const hapticConfig = {config_json};

// Active haptic actuators
let leftActuator = null;
let rightActuator = null;

function initHaptics(inputSources) {{
    for (const inputSource of inputSources) {{
        if (inputSource.gamepad && inputSource.gamepad.hapticActuators) {{
            const actuator = inputSource.gamepad.hapticActuators[0];
            if (inputSource.handedness === 'left') {{
                leftActuator = actuator;
            }} else if (inputSource.handedness === 'right') {{
                rightActuator = actuator;
            }}
        }}
    }}
}}

async function playHapticWaveform(waveform, channel) {{
    if (!hapticConfig.enabled) return;
    
    const actuators = [];
    if (channel === 'left' && leftActuator) actuators.push(leftActuator);
    if (channel === 'right' && rightActuator) actuators.push(rightActuator);
    if (channel === 'both') {{
        if (leftActuator) actuators.push(leftActuator);
        if (rightActuator) actuators.push(rightActuator);
    }}
    
    for (const actuator of actuators) {{
        if (actuator.playEffect) {{
            // GamepadHapticActuator
            await playWaveformOnActuator(actuator, waveform);
        }} else if (actuator.pulse) {{
            // Legacy pulse API
            await playWaveformWithPulse(actuator, waveform);
        }}
    }}
}}

async function playWaveformOnActuator(actuator, waveform) {{
    for (const pulse of waveform.pulses) {{
        await actuator.playEffect('dual-rumble', {{
            duration: pulse.duration,
            strongMagnitude: pulse.intensity,
            weakMagnitude: pulse.intensity * 0.5
        }});
        
        // If not looping, wait for pulse to complete
        if (!waveform.loop) {{
            await sleep(pulse.duration);
        }}
    }}
    
    if (waveform.loop) {{
        // Recursively play for looping waveforms
        requestAnimationFrame(() => playWaveformOnActuator(actuator, waveform));
    }}
}}

async function playWaveformWithPulse(actuator, waveform) {{
    for (const pulse of waveform.pulses) {{
        await actuator.pulse(pulse.intensity, pulse.duration);
        if (!waveform.loop) {{
            await sleep(pulse.duration);
        }}
    }}
}}

function sleep(ms) {{
    return new Promise(resolve => setTimeout(resolve, ms));
}}

// Predefined patterns
const hapticPatterns = {{
    atomSelect: {{
        pulses: [
            {{ intensity: 0.8, duration: 50 }},
            {{ intensity: 0.0, duration: 30 }},
            {{ intensity: 0.4, duration: 30 }}
        ],
        loop: false
    }},
    atomHover: {{
        pulses: [{{ intensity: 0.2, duration: 20 }}],
        loop: false
    }},
    bondForm: {{
        pulses: [
            {{ intensity: 0.3, duration: 30 }},
            {{ intensity: 0.6, duration: 50 }},
            {{ intensity: 0.9, duration: 80 }},
            {{ intensity: 0.5, duration: 40 }}
        ],
        loop: false
    }},
    collision: {{
        pulses: [{{ intensity: 1.0, duration: 100 }}],
        loop: false
    }},
    repulsion: {{
        pulses: [
            {{ intensity: 0.6, duration: 20 }},
            {{ intensity: 0.4, duration: 20 }},
            {{ intensity: 0.2, duration: 20 }}
        ],
        loop: true
    }},
    attraction: {{
        pulses: [
            {{ intensity: 0.2, duration: 50 }},
            {{ intensity: 0.4, duration: 50 }},
            {{ intensity: 0.6, duration: 50 }}
        ],
        loop: true
    }}
}};

// Trigger haptic feedback from molecular force
function hapticFromForce(force, hand) {{
    const magnitude = Math.sqrt(force[0]**2 + force[1]**2 + force[2]**2);
    const intensity = Math.min(magnitude / hapticConfig.forceCutoff, 1.0) * hapticConfig.forceFeedbackScale;
    
    if (intensity < hapticConfig.minIntensity) return;
    
    const channel = hand || 'right';
    const actuator = channel === 'left' ? leftActuator : rightActuator;
    
    if (actuator && actuator.pulse) {{
        actuator.pulse(intensity * hapticConfig.intensityScale, 16);  // 60fps
    }}
}}

// Trigger pattern by name
function triggerHapticPattern(patternName, channel = 'both') {{
    const pattern = hapticPatterns[patternName];
    if (pattern) {{
        playHapticWaveform(pattern, channel);
    }}
}}

// Handle molecular interaction feedback
function onMolecularHaptic(eventType, eventData, hand) {{
    const patternMap = {{
        'atom_selected': 'atomSelect',
        'atom_hovered': 'atomHover',
        'bond_formed': 'bondForm',
        'collision_detected': 'collision'
    }};
    
    const patternName = patternMap[eventType];
    if (patternName) {{
        triggerHapticPattern(patternName, hand);
    }}
}}

// Continuous force feedback update
let forceFeedbackInterval = null;

function startForceFeedback() {{
    if (forceFeedbackInterval) return;
    
    forceFeedbackInterval = setInterval(() => {{
        // Get current forces from molecular simulation
        const forces = getMolecularForces();
        
        if (forces.left) {{
            hapticFromForce(forces.left, 'left');
        }}
        if (forces.right) {{
            hapticFromForce(forces.right, 'right');
        }}
    }}, 1000 / hapticConfig.continuousUpdateRate);
}}

function stopForceFeedback() {{
    if (forceFeedbackInterval) {{
        clearInterval(forceFeedbackInterval);
        forceFeedbackInterval = null;
    }}
}}
"""
