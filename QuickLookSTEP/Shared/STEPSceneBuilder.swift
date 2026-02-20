import SceneKit

class STEPSceneBuilder {

    // MARK: - Style Constants

    enum Style {
        static let backgroundColor = NSColor(red: 0.12, green: 0.14, blue: 0.19, alpha: 1.0)
        static let materialColor = NSColor(red: 0.75, green: 0.75, blue: 0.78, alpha: 1.0)
        static let materialMetalness: CGFloat = 0.05
        static let materialRoughness: CGFloat = 0.8
        static let cameraDistanceMultiplier: CGFloat = 2.0
        static let keyLightIntensity: CGFloat = 800
        static let fillLightIntensity: CGFloat = 400
        static let ambientLightIntensity: CGFloat = 300
    }

    // MARK: - Errors

    enum STEPError: Error {
        case parseFailed
        case emptyMesh
    }

    // MARK: - Public API

    static func buildScene(fromPath path: String) throws -> SCNScene {
        let geometry = try buildGeometry(fromPath: path)
        geometry.materials = [buildMaterial()]

        let scene = SCNScene()
        let modelNode = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(modelNode)

        setupCamera(for: modelNode, in: scene)
        setupLighting(for: modelNode, in: scene)
        scene.background.contents = Style.backgroundColor

        return scene
    }

    // MARK: - Geometry

    private static func buildGeometry(fromPath path: String) throws -> SCNGeometry {
        guard let meshPtr = foxtrot_parse_step_file(path) else {
            throw STEPError.parseFailed
        }
        defer { foxtrot_free_mesh(meshPtr) }

        let mesh = meshPtr.pointee
        guard mesh.vertex_count > 0 && mesh.index_count > 0 else {
            throw STEPError.emptyMesh
        }

        let vertexCount = Int(mesh.vertex_count)
        let indexCount = Int(mesh.index_count)

        var positions = [SCNVector3]()
        var normals = [SCNVector3]()
        positions.reserveCapacity(vertexCount)
        normals.reserveCapacity(vertexCount)

        for i in 0..<vertexCount {
            let v = mesh.vertices[i]
            positions.append(SCNVector3(v.x, v.y, v.z))
            normals.append(SCNVector3(v.nx, v.ny, v.nz))
        }

        let positionSource = SCNGeometrySource(vertices: positions)
        let normalSource = SCNGeometrySource(normals: normals)

        let indexData = Data(bytes: mesh.indices, count: indexCount * MemoryLayout<UInt32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indexCount / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )

        return SCNGeometry(sources: [positionSource, normalSource], elements: [element])
    }

    // MARK: - Material

    private static func buildMaterial() -> SCNMaterial {
        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.diffuse.contents = Style.materialColor
        material.metalness.contents = NSColor(white: Style.materialMetalness, alpha: 1.0)
        material.roughness.contents = NSColor(white: Style.materialRoughness, alpha: 1.0)
        material.isDoubleSided = true
        return material
    }

    // MARK: - Camera

    private static func setupCamera(for modelNode: SCNNode, in scene: SCNScene) {
        let (minBound, maxBound) = modelNode.boundingBox
        let center = SCNVector3(
            (minBound.x + maxBound.x) / 2,
            (minBound.y + maxBound.y) / 2,
            (minBound.z + maxBound.z) / 2
        )
        let size = SCNVector3(
            maxBound.x - minBound.x,
            maxBound.y - minBound.y,
            maxBound.z - minBound.z
        )
        let maxDim = max(size.x, max(size.y, size.z))

        modelNode.position = SCNVector3(-center.x, -center.y, -center.z)

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.automaticallyAdjustsZRange = true
        cameraNode.position = SCNVector3(0, 0, maxDim * Style.cameraDistanceMultiplier)
        cameraNode.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(cameraNode)
    }

    // MARK: - Lighting

    private static func setupLighting(for modelNode: SCNNode, in scene: SCNScene) {
        let (minBound, maxBound) = modelNode.boundingBox
        let size = SCNVector3(
            maxBound.x - minBound.x,
            maxBound.y - minBound.y,
            maxBound.z - minBound.z
        )
        let maxDim = max(size.x, max(size.y, size.z))

        let keyLight = SCNNode()
        keyLight.light = SCNLight()
        keyLight.light?.type = .directional
        keyLight.light?.intensity = Style.keyLightIntensity
        keyLight.light?.color = NSColor.white
        keyLight.position = SCNVector3(maxDim, maxDim, maxDim * 2)
        keyLight.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(keyLight)

        let fillLight = SCNNode()
        fillLight.light = SCNLight()
        fillLight.light?.type = .directional
        fillLight.light?.intensity = Style.fillLightIntensity
        fillLight.light?.color = NSColor.white
        fillLight.position = SCNVector3(-maxDim, -maxDim * 0.5, maxDim)
        fillLight.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(fillLight)

        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.intensity = Style.ambientLightIntensity
        ambientLight.light?.color = NSColor(white: 0.5, alpha: 1.0)
        scene.rootNode.addChildNode(ambientLight)
    }
}
