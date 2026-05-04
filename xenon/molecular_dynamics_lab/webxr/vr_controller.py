"""WebXR VR Controller for immersive molecular visualization.

Provides VR-ready WebXR support for molecular dynamics visualization
including hand tracking, controller input, and spatial interactions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class XRSessionMode(Enum):
    """WebXR session modes."""

    INLINE = "inline"
    IMMERSIVE_VR = "immersive-vr"
    IMMERSIVE_AR = "immersive-ar"


class XRReferenceSpace(Enum):
    """WebXR reference space types."""

    VIEWER = "viewer"
    LOCAL = "local"
    LOCAL_FLOOR = "local-floor"
    BOUNDED_FLOOR = "bounded-floor"
    UNBOUNDED = "unbounded"


class HandJoint(Enum):
    """Hand tracking joint names."""

    WRIST = "wrist"
    THUMB_METACARPAL = "thumb-metacarpal"
    THUMB_PHALANX_PROXIMAL = "thumb-phalanx-proximal"
    THUMB_PHALANX_DISTAL = "thumb-phalanx-distal"
    THUMB_TIP = "thumb-tip"
    INDEX_FINGER_METACARPAL = "index-finger-metacarpal"
    INDEX_FINGER_PHALANX_PROXIMAL = "index-finger-phalanx-proximal"
    INDEX_FINGER_PHALANX_INTERMEDIATE = "index-finger-phalanx-intermediate"
    INDEX_FINGER_PHALANX_DISTAL = "index-finger-phalanx-distal"
    INDEX_FINGER_TIP = "index-finger-tip"
    MIDDLE_FINGER_TIP = "middle-finger-tip"
    RING_FINGER_TIP = "ring-finger-tip"
    PINKY_FINGER_TIP = "pinky-finger-tip"


class ControllerButton(Enum):
    """VR controller button names."""

    TRIGGER = "trigger"
    GRIP = "grip"
    THUMBSTICK = "thumbstick"
    THUMBSTICK_CLICK = "thumbstick-click"
    A_BUTTON = "a-button"
    B_BUTTON = "b-button"
    X_BUTTON = "x-button"
    Y_BUTTON = "y-button"
    MENU = "menu"


@dataclass
class VRConfig:
    """Configuration for VR controller."""

    session_mode: XRSessionMode = XRSessionMode.IMMERSIVE_VR
    reference_space: XRReferenceSpace = XRReferenceSpace.LOCAL_FLOOR
    enable_hand_tracking: bool = True
    enable_controllers: bool = True
    interaction_radius: float = 0.1  # meters
    selection_highlight_color: str = "#00ff00"
    molecule_scale: float = 0.1  # meters per Angstrom
    movement_speed: float = 1.0  # meters per second
    rotation_speed: float = 45.0  # degrees per second
    snap_to_atom: bool = True
    show_atom_labels: bool = True
    show_bond_distances: bool = True
    enable_voice_commands: bool = False
    render_quality: str = "high"  # low, medium, high

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sessionMode": self.session_mode.value,
            "referenceSpace": self.reference_space.value,
            "enableHandTracking": self.enable_hand_tracking,
            "enableControllers": self.enable_controllers,
            "interactionRadius": self.interaction_radius,
            "selectionHighlightColor": self.selection_highlight_color,
            "moleculeScale": self.molecule_scale,
            "movementSpeed": self.movement_speed,
            "rotationSpeed": self.rotation_speed,
            "snapToAtom": self.snap_to_atom,
            "showAtomLabels": self.show_atom_labels,
            "showBondDistances": self.show_bond_distances,
            "enableVoiceCommands": self.enable_voice_commands,
            "renderQuality": self.render_quality,
        }


@dataclass
class ControllerState:
    """State of a VR controller."""

    connected: bool = False
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # quaternion
    buttons: dict[str, float] = field(default_factory=dict)
    thumbstick: tuple[float, float] = (0.0, 0.0)
    handedness: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "connected": self.connected,
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "buttons": self.buttons,
            "thumbstick": list(self.thumbstick),
            "handedness": self.handedness,
        }


@dataclass
class HandState:
    """State of hand tracking."""

    detected: bool = False
    joints: dict[str, np.ndarray] = field(default_factory=dict)
    pinch_strength: float = 0.0
    grip_strength: float = 0.0
    handedness: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "detected": self.detected,
            "joints": {k: v.tolist() for k, v in self.joints.items()},
            "pinchStrength": self.pinch_strength,
            "gripStrength": self.grip_strength,
            "handedness": self.handedness,
        }


@dataclass
class XRInteraction:
    """Represents an XR interaction event."""

    interaction_type: str  # select, squeeze, hover, etc.
    hand: str  # left, right
    target_type: str  # atom, bond, residue, surface, etc.
    target_id: Optional[int] = None
    position: Optional[np.ndarray] = None
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "interactionType": self.interaction_type,
            "hand": self.hand,
            "targetType": self.target_type,
            "targetId": self.target_id,
            "position": self.position.tolist() if self.position is not None else None,
            "timestamp": self.timestamp,
        }


class VRController:
    """WebXR VR Controller for immersive molecular visualization.

    Manages VR session, controller/hand input, and molecular
    interactions in virtual reality.
    """

    def __init__(self, config: Optional[VRConfig] = None):
        """Initialize VR controller.

        Args:
            config: VR configuration
        """
        self.config = config or VRConfig()
        self._session_active = False
        self._left_controller = ControllerState(handedness="left")
        self._right_controller = ControllerState(handedness="right")
        self._left_hand = HandState(handedness="left")
        self._right_hand = HandState(handedness="right")
        self._selected_atoms: list[int] = []
        self._hovered_atom: Optional[int] = None
        self._grabbed_atom: Optional[int] = None
        self._interaction_callbacks: list[Callable[[XRInteraction], None]] = []
        self._molecule_transform = np.eye(4)

    def is_supported(self) -> str:
        """Generate JavaScript to check WebXR support.

        Returns:
            JavaScript code for support check
        """
        return """
async function checkXRSupport() {
    if (!navigator.xr) {
        return { supported: false, reason: 'WebXR not available' };
    }
    
    const modes = ['immersive-vr', 'immersive-ar', 'inline'];
    const support = {};
    
    for (const mode of modes) {
        try {
            support[mode] = await navigator.xr.isSessionSupported(mode);
        } catch (e) {
            support[mode] = false;
        }
    }
    
    return {
        supported: support['immersive-vr'] || support['immersive-ar'],
        modes: support
    };
}
"""

    def generate_session_init_js(self) -> str:
        """Generate JavaScript for XR session initialization.

        Returns:
            JavaScript code for session init
        """
        config_json = json.dumps(self.config.to_dict())

        return f"""
// VR Session Configuration
const vrConfig = {config_json};

let xrSession = null;
let xrRefSpace = null;
let xrFrame = null;
let leftController = null;
let rightController = null;

async function initXRSession() {{
    if (!navigator.xr) {{
        console.error('WebXR not supported');
        return null;
    }}
    
    try {{
        const sessionInit = {{
            requiredFeatures: [vrConfig.referenceSpace],
            optionalFeatures: ['hand-tracking', 'local-floor']
        }};
        
        xrSession = await navigator.xr.requestSession(vrConfig.sessionMode, sessionInit);
        
        // Set up reference space
        xrRefSpace = await xrSession.requestReferenceSpace(vrConfig.referenceSpace);
        
        // Set up input sources
        xrSession.addEventListener('inputsourceschange', onInputSourcesChange);
        xrSession.addEventListener('selectstart', onSelectStart);
        xrSession.addEventListener('selectend', onSelectEnd);
        xrSession.addEventListener('squeezestart', onSqueezeStart);
        xrSession.addEventListener('squeezeend', onSqueezeEnd);
        
        console.log('XR Session initialized');
        return xrSession;
        
    }} catch (error) {{
        console.error('Failed to initialize XR session:', error);
        return null;
    }}
}}

function onInputSourcesChange(event) {{
    for (const inputSource of event.added) {{
        if (inputSource.handedness === 'left') {{
            leftController = inputSource;
        }} else if (inputSource.handedness === 'right') {{
            rightController = inputSource;
        }}
    }}
    
    for (const inputSource of event.removed) {{
        if (inputSource === leftController) leftController = null;
        if (inputSource === rightController) rightController = null;
    }}
}}

function onSelectStart(event) {{
    const inputSource = event.inputSource;
    const targetRay = xrFrame.getPose(inputSource.targetRaySpace, xrRefSpace);
    
    if (targetRay) {{
        // Ray-cast to find atom
        const hitAtom = raycastToAtom(targetRay.transform.position, targetRay.transform.orientation);
        if (hitAtom !== null) {{
            selectAtom(hitAtom, inputSource.handedness);
        }}
    }}
}}

function onSelectEnd(event) {{
    // Handle selection end
}}

function onSqueezeStart(event) {{
    const inputSource = event.inputSource;
    const grip = xrFrame.getPose(inputSource.gripSpace, xrRefSpace);
    
    if (grip) {{
        // Check if grabbing an atom
        const nearbyAtom = findNearbyAtom(grip.transform.position, vrConfig.interactionRadius);
        if (nearbyAtom !== null) {{
            grabAtom(nearbyAtom, inputSource.handedness);
        }}
    }}
}}

function onSqueezeEnd(event) {{
    releaseAtom(event.inputSource.handedness);
}}

function updateControllerPoses(frame, refSpace) {{
    xrFrame = frame;
    
    if (leftController) {{
        const leftPose = frame.getPose(leftController.gripSpace, refSpace);
        if (leftPose) {{
            updateControllerVisual('left', leftPose.transform);
        }}
        
        // Hand tracking
        if (leftController.hand) {{
            const leftHand = frame.getJointPose(leftController.hand.get('index-finger-tip'), refSpace);
            // Update hand visual
        }}
    }}
    
    if (rightController) {{
        const rightPose = frame.getPose(rightController.gripSpace, refSpace);
        if (rightPose) {{
            updateControllerVisual('right', rightPose.transform);
        }}
    }}
}}

// Interaction functions
function selectAtom(atomIndex, hand) {{
    const interaction = {{
        type: 'select',
        hand: hand,
        atomIndex: atomIndex,
        timestamp: performance.now()
    }};
    
    onMolecularInteraction(interaction);
    highlightAtom(atomIndex, vrConfig.selectionHighlightColor);
}}

function grabAtom(atomIndex, hand) {{
    grabbedAtom = atomIndex;
    grabbedHand = hand;
    
    const interaction = {{
        type: 'grab',
        hand: hand,
        atomIndex: atomIndex,
        timestamp: performance.now()
    }};
    
    onMolecularInteraction(interaction);
}}

function releaseAtom(hand) {{
    if (grabbedHand === hand) {{
        const interaction = {{
            type: 'release',
            hand: hand,
            atomIndex: grabbedAtom,
            timestamp: performance.now()
        }};
        
        onMolecularInteraction(interaction);
        grabbedAtom = null;
        grabbedHand = null;
    }}
}}

function raycastToAtom(origin, direction) {{
    // Implement ray-atom intersection
    // Returns atom index or null
    const atoms = getMoleculeAtoms();
    let closestAtom = null;
    let closestDist = Infinity;
    
    for (let i = 0; i < atoms.length; i++) {{
        const atom = atoms[i];
        const atomPos = new THREE.Vector3(atom.x, atom.y, atom.z).multiplyScalar(vrConfig.moleculeScale);
        const atomRadius = getAtomRadius(atom.element) * vrConfig.moleculeScale;
        
        // Ray-sphere intersection
        const dist = rayIntersectSphere(origin, direction, atomPos, atomRadius);
        if (dist !== null && dist < closestDist) {{
            closestDist = dist;
            closestAtom = i;
        }}
    }}
    
    return closestAtom;
}}

function findNearbyAtom(position, radius) {{
    const atoms = getMoleculeAtoms();
    let closestAtom = null;
    let closestDist = radius;
    
    for (let i = 0; i < atoms.length; i++) {{
        const atom = atoms[i];
        const atomPos = new THREE.Vector3(atom.x, atom.y, atom.z).multiplyScalar(vrConfig.moleculeScale);
        const dist = position.distanceTo(atomPos);
        
        if (dist < closestDist) {{
            closestDist = dist;
            closestAtom = i;
        }}
    }}
    
    return closestAtom;
}}

function onMolecularInteraction(interaction) {{
    // Send to Python backend via WebSocket
    if (window.molecularLabSocket && window.molecularLabSocket.readyState === WebSocket.OPEN) {{
        window.molecularLabSocket.send(JSON.stringify({{
            type: 'interaction',
            data: interaction
        }}));
    }}
    
    console.log('Molecular interaction:', interaction);
}}
"""

    def generate_hand_tracking_js(self) -> str:
        """Generate JavaScript for hand tracking.

        Returns:
            JavaScript code for hand tracking
        """
        return """
// Hand tracking state
const handState = {
    left: { detected: false, joints: {}, pinchStrength: 0, gripStrength: 0 },
    right: { detected: false, joints: {}, pinchStrength: 0, gripStrength: 0 }
};

const HAND_JOINTS = [
    'wrist',
    'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
    'index-finger-metacarpal', 'index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate', 'index-finger-phalanx-distal', 'index-finger-tip',
    'middle-finger-metacarpal', 'middle-finger-phalanx-proximal', 'middle-finger-phalanx-intermediate', 'middle-finger-phalanx-distal', 'middle-finger-tip',
    'ring-finger-metacarpal', 'ring-finger-phalanx-proximal', 'ring-finger-phalanx-intermediate', 'ring-finger-phalanx-distal', 'ring-finger-tip',
    'pinky-finger-metacarpal', 'pinky-finger-phalanx-proximal', 'pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-distal', 'pinky-finger-tip'
];

function updateHandTracking(frame, refSpace, inputSource) {
    if (!inputSource.hand) return;
    
    const hand = inputSource.handedness;
    const joints = {};
    
    for (const jointName of HAND_JOINTS) {
        const joint = inputSource.hand.get(jointName);
        if (joint) {
            const pose = frame.getJointPose(joint, refSpace);
            if (pose) {
                joints[jointName] = {
                    position: [pose.transform.position.x, pose.transform.position.y, pose.transform.position.z],
                    orientation: [pose.transform.orientation.x, pose.transform.orientation.y, pose.transform.orientation.z, pose.transform.orientation.w],
                    radius: pose.radius
                };
            }
        }
    }
    
    handState[hand].detected = Object.keys(joints).length > 0;
    handState[hand].joints = joints;
    
    // Calculate pinch strength (thumb tip to index finger tip distance)
    if (joints['thumb-tip'] && joints['index-finger-tip']) {
        const thumbTip = new THREE.Vector3(...joints['thumb-tip'].position);
        const indexTip = new THREE.Vector3(...joints['index-finger-tip'].position);
        const pinchDist = thumbTip.distanceTo(indexTip);
        handState[hand].pinchStrength = Math.max(0, 1 - pinchDist / 0.05); // 5cm = no pinch
    }
    
    // Calculate grip strength (average curl of fingers)
    const fingerTips = ['index-finger-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip'];
    const fingerBases = ['index-finger-metacarpal', 'middle-finger-metacarpal', 'ring-finger-metacarpal', 'pinky-finger-metacarpal'];
    
    let totalCurl = 0;
    let fingerCount = 0;
    
    for (let i = 0; i < fingerTips.length; i++) {
        if (joints[fingerTips[i]] && joints[fingerBases[i]] && joints['wrist']) {
            const tip = new THREE.Vector3(...joints[fingerTips[i]].position);
            const base = new THREE.Vector3(...joints[fingerBases[i]].position);
            const wrist = new THREE.Vector3(...joints['wrist'].position);
            
            const baseToTip = tip.distanceTo(base);
            const baseToWrist = wrist.distanceTo(base);
            
            // Curl is how close tip is to wrist relative to extended position
            totalCurl += Math.max(0, 1 - baseToTip / (baseToWrist * 1.5));
            fingerCount++;
        }
    }
    
    handState[hand].gripStrength = fingerCount > 0 ? totalCurl / fingerCount : 0;
    
    // Check for gestures
    checkHandGestures(hand);
    
    // Render hand skeleton
    renderHandSkeleton(hand, joints);
}

function checkHandGestures(hand) {
    const state = handState[hand];
    
    // Pinch gesture
    if (state.pinchStrength > 0.8) {
        onHandGesture(hand, 'pinch', state.pinchStrength);
    }
    
    // Grip gesture
    if (state.gripStrength > 0.7) {
        onHandGesture(hand, 'grip', state.gripStrength);
    }
    
    // Point gesture (index extended, others curled)
    if (state.joints['index-finger-tip'] && state.pinchStrength < 0.3) {
        const indexTip = new THREE.Vector3(...state.joints['index-finger-tip'].position);
        const indexBase = new THREE.Vector3(...state.joints['index-finger-metacarpal'].position);
        
        // Raycast from finger
        const direction = indexTip.clone().sub(indexBase).normalize();
        const hitAtom = raycastToAtom(indexTip, direction);
        
        if (hitAtom !== null) {
            hoverAtom(hitAtom);
        }
    }
}

function onHandGesture(hand, gesture, strength) {
    const interaction = {
        type: 'gesture',
        hand: hand,
        gesture: gesture,
        strength: strength,
        timestamp: performance.now()
    };
    
    // Handle pinch to select
    if (gesture === 'pinch' && strength > 0.9) {
        const tip = handState[hand].joints['index-finger-tip'];
        if (tip) {
            const position = new THREE.Vector3(...tip.position);
            const nearbyAtom = findNearbyAtom(position, vrConfig.interactionRadius);
            if (nearbyAtom !== null) {
                selectAtom(nearbyAtom, hand);
            }
        }
    }
    
    // Handle grip to grab
    if (gesture === 'grip' && strength > 0.8) {
        const wrist = handState[hand].joints['wrist'];
        if (wrist) {
            const position = new THREE.Vector3(...wrist.position);
            const nearbyAtom = findNearbyAtom(position, vrConfig.interactionRadius * 2);
            if (nearbyAtom !== null) {
                grabAtom(nearbyAtom, hand);
            }
        }
    }
    
    onMolecularInteraction(interaction);
}

function renderHandSkeleton(hand, joints) {
    // Create/update hand skeleton visualization
    const color = hand === 'left' ? 0x00ff00 : 0xff0000;
    
    // Render joints as spheres
    for (const [jointName, jointData] of Object.entries(joints)) {
        const sphere = getOrCreateJointSphere(`${hand}-${jointName}`);
        sphere.position.set(...jointData.position);
        sphere.material.color.setHex(color);
        sphere.visible = true;
    }
    
    // Render bones as cylinders connecting joints
    const connections = [
        ['wrist', 'thumb-metacarpal'],
        ['thumb-metacarpal', 'thumb-phalanx-proximal'],
        ['thumb-phalanx-proximal', 'thumb-phalanx-distal'],
        ['thumb-phalanx-distal', 'thumb-tip'],
        ['wrist', 'index-finger-metacarpal'],
        ['index-finger-metacarpal', 'index-finger-phalanx-proximal'],
        ['index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate'],
        ['index-finger-phalanx-intermediate', 'index-finger-phalanx-distal'],
        ['index-finger-phalanx-distal', 'index-finger-tip'],
        // ... other fingers
    ];
    
    for (const [joint1, joint2] of connections) {
        if (joints[joint1] && joints[joint2]) {
            renderBone(`${hand}-${joint1}-${joint2}`, joints[joint1].position, joints[joint2].position, color);
        }
    }
}
"""

    def generate_vr_controls_ui(self) -> str:
        """Generate HTML/CSS for VR control overlay.

        Returns:
            HTML string for VR controls
        """
        return """
<div id="vr-controls" style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
    <button id="vr-button" onclick="toggleVR()" style="
        padding: 15px 30px;
        font-size: 18px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    ">
        <span id="vr-icon">🥽</span> Enter VR
    </button>
    
    <div id="vr-status" style="
        margin-top: 10px;
        padding: 10px;
        background: rgba(0,0,0,0.7);
        border-radius: 8px;
        color: white;
        font-size: 14px;
        display: none;
    ">
        <div>VR Status: <span id="vr-session-status">Inactive</span></div>
        <div>Left Hand: <span id="vr-left-status">Not detected</span></div>
        <div>Right Hand: <span id="vr-right-status">Not detected</span></div>
        <div>Selected Atoms: <span id="vr-selected-count">0</span></div>
    </div>
</div>

<style>
    #vr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    #vr-button:active {
        transform: translateY(0);
    }
    
    #vr-button.active {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
        100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }
    
    #vr-button.active {
        animation: pulse 2s infinite;
    }
</style>

<script>
let vrActive = false;

async function toggleVR() {
    const button = document.getElementById('vr-button');
    const statusDiv = document.getElementById('vr-status');
    
    if (!vrActive) {
        const session = await initXRSession();
        if (session) {
            vrActive = true;
            button.classList.add('active');
            button.innerHTML = '<span>🔴</span> Exit VR';
            statusDiv.style.display = 'block';
            document.getElementById('vr-session-status').textContent = 'Active';
        }
    } else {
        if (xrSession) {
            await xrSession.end();
        }
        vrActive = false;
        button.classList.remove('active');
        button.innerHTML = '<span>🥽</span> Enter VR';
        statusDiv.style.display = 'none';
    }
}

function updateVRStatus() {
    document.getElementById('vr-left-status').textContent = 
        leftController ? 'Connected' : (handState.left.detected ? 'Hand Tracking' : 'Not detected');
    document.getElementById('vr-right-status').textContent = 
        rightController ? 'Connected' : (handState.right.detected ? 'Hand Tracking' : 'Not detected');
    document.getElementById('vr-selected-count').textContent = selectedAtoms.length;
}
</script>
"""

    def process_interaction(self, interaction_data: dict) -> dict:
        """Process interaction from VR client.

        Args:
            interaction_data: Interaction event data

        Returns:
            Response data
        """
        interaction = XRInteraction(
            interaction_type=interaction_data.get("type", "unknown"),
            hand=interaction_data.get("hand", "unknown"),
            target_type=interaction_data.get("targetType", "atom"),
            target_id=interaction_data.get("atomIndex"),
            position=np.array(interaction_data.get("position", [0, 0, 0])),
            timestamp=interaction_data.get("timestamp", 0),
        )

        # Call registered callbacks
        for callback in self._interaction_callbacks:
            callback(interaction)

        # Process interaction based on type
        if interaction.interaction_type == "select":
            if interaction.target_id is not None:
                if interaction.target_id not in self._selected_atoms:
                    self._selected_atoms.append(interaction.target_id)
                else:
                    self._selected_atoms.remove(interaction.target_id)

        elif interaction.interaction_type == "grab":
            self._grabbed_atom = interaction.target_id

        elif interaction.interaction_type == "release":
            self._grabbed_atom = None

        return {
            "status": "ok",
            "selectedAtoms": self._selected_atoms,
            "grabbedAtom": self._grabbed_atom,
        }

    def add_interaction_callback(self, callback: Callable[[XRInteraction], None]) -> None:
        """Add callback for VR interactions.

        Args:
            callback: Function called on interaction
        """
        self._interaction_callbacks.append(callback)

    def get_controller_state(self, hand: str) -> ControllerState:
        """Get controller state.

        Args:
            hand: 'left' or 'right'

        Returns:
            ControllerState for specified hand
        """
        return self._left_controller if hand == "left" else self._right_controller

    def get_hand_state(self, hand: str) -> HandState:
        """Get hand tracking state.

        Args:
            hand: 'left' or 'right'

        Returns:
            HandState for specified hand
        """
        return self._left_hand if hand == "left" else self._right_hand

    def get_selected_atoms(self) -> list[int]:
        """Get list of selected atom indices.

        Returns:
            List of atom indices
        """
        return self._selected_atoms.copy()

    def clear_selection(self) -> None:
        """Clear atom selection."""
        self._selected_atoms.clear()
