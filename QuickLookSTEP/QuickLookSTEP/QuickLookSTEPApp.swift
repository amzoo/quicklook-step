import SwiftUI

@main
struct QuickLookSTEPApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 64))
                .foregroundColor(.accentColor)

            Text("QuickLook STEP")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("Preview STEP 3D model files directly in Finder.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Text("Press Space on any .step or .stp file in Finder to see a 3D preview.")
                .font(.callout)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
        .frame(width: 400, height: 300)
        .padding()
    }
}
