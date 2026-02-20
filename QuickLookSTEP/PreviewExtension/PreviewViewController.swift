import AppKit
import QuickLookUI
import SceneKit

class PreviewViewController: NSViewController, QLPreviewingController {

    func preparePreviewOfFile(at url: URL, completionHandler handler: @escaping (Error?) -> Void) {
        do {
            let scene = try STEPSceneBuilder.buildScene(fromPath: url.path)

            let scnView = SCNView(frame: self.view.bounds)
            scnView.autoresizingMask = [.width, .height]
            scnView.scene = scene
            scnView.allowsCameraControl = true
            scnView.autoenablesDefaultLighting = false
            scnView.backgroundColor = STEPSceneBuilder.Style.backgroundColor
            self.view.addSubview(scnView)

            handler(nil)
        } catch {
            handler(error)
        }
    }
}
