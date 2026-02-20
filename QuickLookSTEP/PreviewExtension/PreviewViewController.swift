import Cocoa
import QuickLookUI
import SceneKit

class PreviewViewController: NSViewController, QLPreviewingController {
    override func viewDidLoad() {
        super.viewDidLoad()
    }

    func preparePreviewOfFile(at url: URL, completionHandler handler: @escaping (Error?) -> Void) {
        do {
            let scene = try STEPSceneBuilder.buildScene(fromPath: url.path)

            let scnView = SCNView(frame: self.view.bounds)
            scnView.autoresizingMask = [.width, .height]
            scnView.scene = scene
            scnView.allowsCameraControl = true
            scnView.autoenablesDefaultLighting = false
            scnView.backgroundColor = NSColor(red: 0.12, green: 0.14, blue: 0.19, alpha: 1.0)
            self.view.addSubview(scnView)

            handler(nil)
        } catch {
            handler(error)
        }
    }
}
