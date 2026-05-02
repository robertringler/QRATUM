// QuASIM Technical Portal - Three.js 3D Scenes
// Bloch Sphere, Quantum Circuits, and Tensor Networks

(function() {
  'use strict';

  // Wait for Three.js to load
  if (typeof THREE === 'undefined') {
    console.warn('Three.js not loaded. 3D visualizations will not render.');
    return;
  }

  window.QuASIM = window.QuASIM || {};

  // ============================================================================
  // Bloch Sphere 3D
  // ============================================================================

  window.QuASIM.BlochSphere3D = class {
    constructor(container) {
      this.container = container;
      this.width = container.clientWidth;
      this.height = container.clientHeight;

      this.init();
      this.animate();
      window.addEventListener('resize', () => this.onWindowResize());
    }

    init() {
      // Scene
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x0a0a0f);

      // Camera
      this.camera = new THREE.PerspectiveCamera(
        45,
        this.width / this.height,
        0.1,
        1000
      );
      this.camera.position.set(3, 3, 3);
      this.camera.lookAt(0, 0, 0);

      // Renderer
      this.renderer = new THREE.WebGLRenderer({ antialias: true });
      this.renderer.setSize(this.width, this.height);
      this.container.appendChild(this.renderer.domElement);

      // Controls
      if (typeof THREE.OrbitControls !== 'undefined') {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
      }

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      this.scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(5, 5, 5);
      this.scene.add(directionalLight);

      // Bloch sphere
      const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
      const sphereMaterial = new THREE.MeshPhongMaterial({
        color: 0x1a1a24,
        transparent: true,
        opacity: 0.3,
        wireframe: false
      });
      this.sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
      this.scene.add(this.sphere);

      // Wireframe overlay
      const wireframeGeometry = new THREE.SphereGeometry(1, 16, 16);
      const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ff88,
        wireframe: true,
        transparent: true,
        opacity: 0.2
      });
      const wireframe = new THREE.Mesh(wireframeGeometry, wireframeMaterial);
      this.scene.add(wireframe);

      // Axes
      this.createAxes();

      // State vector
      this.createStateVector();

      // Labels
      this.createLabels();
    }

    createAxes() {
      const material = new THREE.LineBasicMaterial({ color: 0x00ff88 });

      // X axis (red)
      const xGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1.5, 0, 0),
        new THREE.Vector3(1.5, 0, 0)
      ]);
      const xLine = new THREE.Line(xGeometry, new THREE.LineBasicMaterial({ color: 0xff3366 }));
      this.scene.add(xLine);

      // Y axis (green)
      const yGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -1.5, 0),
        new THREE.Vector3(0, 1.5, 0)
      ]);
      const yLine = new THREE.Line(yGeometry, new THREE.LineBasicMaterial({ color: 0x00ff88 }));
      this.scene.add(yLine);

      // Z axis (blue)
      const zGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, -1.5),
        new THREE.Vector3(0, 0, 1.5)
      ]);
      const zLine = new THREE.Line(zGeometry, new THREE.LineBasicMaterial({ color: 0x00d4ff }));
      this.scene.add(zLine);
    }

    createStateVector() {
      // State vector arrow
      const arrowLength = 1;
      const arrowColor = 0xffaa00;

      this.stateArrow = new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(0, 0, 0),
        arrowLength,
        arrowColor,
        0.2,
        0.1
      );
      this.scene.add(this.stateArrow);
    }

    createLabels() {
      // In a real implementation, you'd use CSS2DRenderer or sprites for labels
      // For now, we'll use simple markers

      const markerGeometry = new THREE.SphereGeometry(0.05, 8, 8);

      // |0⟩ state (top)
      const topMarker = new THREE.Mesh(
        markerGeometry,
        new THREE.MeshBasicMaterial({ color: 0x00d4ff })
      );
      topMarker.position.set(0, 0, 1.1);
      this.scene.add(topMarker);

      // |1⟩ state (bottom)
      const bottomMarker = new THREE.Mesh(
        markerGeometry,
        new THREE.MeshBasicMaterial({ color: 0xff3366 })
      );
      bottomMarker.position.set(0, 0, -1.1);
      this.scene.add(bottomMarker);
    }

    updateStateVector(theta, phi) {
      // Update state vector based on angles
      const x = Math.sin(theta) * Math.cos(phi);
      const y = Math.sin(theta) * Math.sin(phi);
      const z = Math.cos(theta);

      this.stateArrow.setDirection(new THREE.Vector3(x, y, z));
    }

    animate() {
      requestAnimationFrame(() => this.animate());

      // Rotate state vector slowly
      const time = Date.now() * 0.001;
      this.updateStateVector(Math.PI / 4, time);

      if (this.controls) {
        this.controls.update();
      }

      this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
      this.width = this.container.clientWidth;
      this.height = this.container.clientHeight;
      this.camera.aspect = this.width / this.height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(this.width, this.height);
    }
  };

  // ============================================================================
  // Quantum Circuit 3D
  // ============================================================================

  window.QuASIM.QuantumCircuit3D = class {
    constructor(container, circuitData) {
      this.container = container;
      this.circuitData = circuitData;
      this.width = container.clientWidth;
      this.height = container.clientHeight;

      this.init();
      this.animate();
      window.addEventListener('resize', () => this.onWindowResize());
    }

    init() {
      // Scene
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x0a0a0f);

      // Camera
      this.camera = new THREE.PerspectiveCamera(
        45,
        this.width / this.height,
        0.1,
        1000
      );
      this.camera.position.set(5, 5, 5);
      this.camera.lookAt(0, 0, 0);

      // Renderer
      this.renderer = new THREE.WebGLRenderer({ antialias: true });
      this.renderer.setSize(this.width, this.height);
      this.container.appendChild(this.renderer.domElement);

      // Controls
      if (typeof THREE.OrbitControls !== 'undefined') {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      }

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      this.scene.add(ambientLight);

      // Create circuit visualization
      this.createCircuit();
    }

    createCircuit() {
      const wireCount = this.circuitData?.wires || 4;
      const gateCount = this.circuitData?.gates || 8;

      // Create wires
      const wireMaterial = new THREE.LineBasicMaterial({ color: 0x00ff88 });

      for (let i = 0; i < wireCount; i++) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-5, i * 0.5, 0),
          new THREE.Vector3(5, i * 0.5, 0)
        ]);
        const wire = new THREE.Line(geometry, wireMaterial);
        this.scene.add(wire);
      }

      // Create gates (as boxes)
      const gateGeometry = new THREE.BoxGeometry(0.4, 0.4, 0.4);
      const gateMaterial = new THREE.MeshPhongMaterial({ color: 0x00d4ff });

      for (let i = 0; i < gateCount; i++) {
        const gate = new THREE.Mesh(gateGeometry, gateMaterial);
        gate.position.set(
          -4 + (i * 1.2),
          Math.floor(Math.random() * wireCount) * 0.5,
          0
        );
        this.scene.add(gate);
      }
    }

    animate() {
      requestAnimationFrame(() => this.animate());

      if (this.controls) {
        this.controls.update();
      }

      this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
      this.width = this.container.clientWidth;
      this.height = this.container.clientHeight;
      this.camera.aspect = this.width / this.height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(this.width, this.height);
    }
  };

  // ============================================================================
  // Tensor Network 3D
  // ============================================================================

  window.QuASIM.TensorNetwork3D = class {
    constructor(container, networkData) {
      this.container = container;
      this.networkData = networkData;
      this.width = container.clientWidth;
      this.height = container.clientHeight;

      this.init();
      this.animate();
      window.addEventListener('resize', () => this.onWindowResize());
    }

    init() {
      // Scene
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x0a0a0f);

      // Camera
      this.camera = new THREE.PerspectiveCamera(
        45,
        this.width / this.height,
        0.1,
        1000
      );
      this.camera.position.set(5, 5, 5);
      this.camera.lookAt(0, 0, 0);

      // Renderer
      this.renderer = new THREE.WebGLRenderer({ antialias: true });
      this.renderer.setSize(this.width, this.height);
      this.container.appendChild(this.renderer.domElement);

      // Controls
      if (typeof THREE.OrbitControls !== 'undefined') {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      }

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      this.scene.add(ambientLight);

      // Create network
      this.createNetwork();
    }

    createNetwork() {
      const nodeCount = this.networkData?.nodes || 12;

      // Create nodes
      const nodeGeometry = new THREE.SphereGeometry(0.2, 16, 16);
      const nodeMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff88 });

      const nodes = [];
      for (let i = 0; i < nodeCount; i++) {
        const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
        const angle = (i / nodeCount) * Math.PI * 2;
        const radius = 2;
        node.position.set(
          Math.cos(angle) * radius,
          (Math.random() - 0.5) * 2,
          Math.sin(angle) * radius
        );
        this.scene.add(node);
        nodes.push(node);
      }

      // Create edges
      const edgeMaterial = new THREE.LineBasicMaterial({
        color: 0x00d4ff,
        transparent: true,
        opacity: 0.5
      });

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          if (Math.random() > 0.7) {
            const geometry = new THREE.BufferGeometry().setFromPoints([
              nodes[i].position,
              nodes[j].position
            ]);
            const edge = new THREE.Line(geometry, edgeMaterial);
            this.scene.add(edge);
          }
        }
      }
    }

    animate() {
      requestAnimationFrame(() => this.animate());

      if (this.controls) {
        this.controls.update();
      }

      this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
      this.width = this.container.clientWidth;
      this.height = this.container.clientHeight;
      this.camera.aspect = this.width / this.height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(this.width, this.height);
    }
  };

  // ============================================================================
  // Auto-initialize 3D scenes
  // ============================================================================

  document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bloch Sphere
    const blochContainer = document.getElementById('bloch-sphere-container');
    if (blochContainer) {
      new window.QuASIM.BlochSphere3D(blochContainer);
    }

    // Initialize Quantum Circuit
    const circuitContainer = document.getElementById('circuit-3d-container');
    if (circuitContainer) {
      new window.QuASIM.QuantumCircuit3D(circuitContainer, { wires: 4, gates: 10 });
    }

    // Initialize Tensor Network
    const tensorContainer = document.getElementById('tensor-network-container');
    if (tensorContainer) {
      new window.QuASIM.TensorNetwork3D(tensorContainer, { nodes: 12 });
    }
  });

})();
