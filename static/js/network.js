class NetworkGraph {
    constructor(elementId, data) {
        this.elementId = elementId;
        this.data = data;
        this.init();
    }

    init() {
        if (typeof THREE === 'undefined') {
            console.error('Three.js is required for NetworkGraph');
            return;
        }

        const container = document.getElementById(this.elementId);
        const width = container.clientWidth;
        const height = container.clientHeight;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true });
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);

        // Add OrbitControls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = false;
        controls.minDistance = 100;
        controls.maxDistance = 500;

        // Nodes
        const nodes = this.data.nodes.map(node => {
            const geometry = new THREE.SphereGeometry(5, 32, 32);
            const material = new THREE.MeshBasicMaterial({ color: 0x00ffea });
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(
                Math.random() * 200 - 100,
                Math.random() * 200 - 100,
                Math.random() * 200 - 100
            );
            scene.add(sphere);
            return { mesh: sphere, id: node.id };
        });

        // Links
        this.data.links.forEach(link => {
            const sourceNode = nodes.find(n => n.id === link.source);
            const targetNode = nodes.find(n => n.id === link.target);
            if (sourceNode && targetNode) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    sourceNode.mesh.position,
                    targetNode.mesh.position
                ]);
                const material = new THREE.LineBasicMaterial({ color: 0xff00ff });
                const line = new THREE.Line(geometry, material);
                scene.add(line);
            }
        });

        camera.position.z = 300;

        const animate = () => {
            requestAnimationFrame(animate);
            nodes.forEach(node => {
                node.mesh.rotation.y += 0.01;
            });
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        window.addEventListener('resize', () => {
            const newWidth = container.clientWidth;
            const newHeight = container.clientHeight;
            renderer.setSize(newWidth, newHeight);
            camera.aspect = newWidth / newHeight;
            camera.updateProjectionMatrix();
        });
    }
}