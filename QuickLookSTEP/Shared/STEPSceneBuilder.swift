import SceneKit
import Foundation

class STEPSceneBuilder {
    enum STEPError: Error {
        case parseFailed
        case emptyMesh
    }

    static func buildScene(fromPath path: String) throws -> SCNScene {
        guard let meshPtr = foxtrot_parse_step_file(path) else {
            throw STEPError.parseFailed
        }
        defer { foxtrot_free_mesh(meshPtr) }

        let mesh = meshPtr.pointee
        guard mesh.vertex_count > 0 && mesh.index_count > 0 else {
            throw STEPError.emptyMesh
        }

        // Build geometry sources
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

        // Index element
        let indexData = Data(bytes: mesh.indices, count: indexCount * MemoryLayout<UInt32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indexCount / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )

        let geometry = SCNGeometry(sources: [positionSource, normalSource], elements: [element])

        // Light gray CAD-style material
        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.diffuse.contents = NSColor(red: 0.75, green: 0.75, blue: 0.78, alpha: 1.0)
        material.metalness.contents = NSColor(white: 0.05, alpha: 1.0)
        material.roughness.contents = NSColor(white: 0.8, alpha: 1.0)
        material.isDoubleSided = true
        geometry.materials = [material]

        // Scene setup
        let scene = SCNScene()
        let modelNode = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(modelNode)

        // Compute bounding box for camera positioning
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

        // Center the model
        modelNode.position = SCNVector3(-center.x, -center.y, -center.z)

        // Camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.automaticallyAdjustsZRange = true
        cameraNode.position = SCNVector3(0, 0, Float(maxDim) * 2.0)
        cameraNode.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(cameraNode)

        // Lighting
        let keyLight = SCNNode()
        keyLight.light = SCNLight()
        keyLight.light?.type = .directional
        keyLight.light?.intensity = 800
        keyLight.light?.color = NSColor.white
        keyLight.position = SCNVector3(Float(maxDim), Float(maxDim), Float(maxDim) * 2)
        keyLight.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(keyLight)

        let fillLight = SCNNode()
        fillLight.light = SCNLight()
        fillLight.light?.type = .directional
        fillLight.light?.intensity = 400
        fillLight.light?.color = NSColor.white
        fillLight.position = SCNVector3(-Float(maxDim), -Float(maxDim) * 0.5, Float(maxDim))
        fillLight.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(fillLight)

        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.intensity = 300
        ambientLight.light?.color = NSColor(white: 0.5, alpha: 1.0)
        scene.rootNode.addChildNode(ambientLight)

        // Background
        scene.background.contents = NSColor(red: 0.12, green: 0.14, blue: 0.19, alpha: 1.0)

        return scene
    }
}
