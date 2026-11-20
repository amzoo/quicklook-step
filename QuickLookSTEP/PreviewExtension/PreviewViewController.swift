import Foundation
import QuickLookUI
import SceneKit

/// Data-based preview. Converts the STEP mesh to a temporary USDZ and hands it
/// to Quick Look, so Apple's own 3D viewer renders it. Unlike a custom SCNView,
/// that viewer keeps click-drag rotation working inside Finder's inline
/// preview pane (Column View), where the host otherwise claims mouse drags.
class PreviewViewController: QLPreviewProvider, QLPreviewingController {

    private static let tempDirPrefix = "quicklook-step-"
    private static let staleAge: TimeInterval = 10 * 60

    func providePreview(for request: QLFilePreviewRequest) async throws -> QLPreviewReply {
        let scene = try STEPSceneBuilder.buildScene(fromPath: request.fileURL.path)

        // Export geometry only. Apple's viewer supplies its own camera and
        // lighting and frames the model to fit; our camera and light nodes
        // would otherwise enlarge the scene bounds and shrink the model.
        for node in scene.rootNode.childNodes where node.camera != nil || node.light != nil {
            node.removeFromParentNode()
        }

        let fm = FileManager.default
        let tmp = fm.temporaryDirectory
        Self.removeStaleTempDirs(in: tmp, using: fm)

        // Quick Look reads the file after we return, so it can't be deleted
        // here; stale directories are purged on the next preview instead.
        let dir = tmp.appendingPathComponent(Self.tempDirPrefix + UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let name = request.fileURL.deletingPathExtension().lastPathComponent
        let usdz = dir.appendingPathComponent(name).appendingPathExtension("usdz")

        guard scene.write(to: usdz, options: nil, delegate: nil, progressHandler: nil) else {
            try? fm.removeItem(at: dir)
            throw CocoaError(.fileWriteUnknown)
        }

        return QLPreviewReply(fileURL: usdz)
    }

    private static func removeStaleTempDirs(in tmp: URL, using fm: FileManager) {
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        guard let items = try? fm.contentsOfDirectory(at: tmp, includingPropertiesForKeys: keys) else { return }
        let cutoff = Date().addingTimeInterval(-staleAge)
        for item in items where item.lastPathComponent.hasPrefix(tempDirPrefix) {
            let modified = (try? item.resourceValues(forKeys: Set(keys)).contentModificationDate) ?? .distantPast
            if modified < cutoff { try? fm.removeItem(at: item) }
        }
    }
}
