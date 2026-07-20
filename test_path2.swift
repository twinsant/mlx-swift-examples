import Foundation

let into = URL(fileURLWithPath: NSHomeDirectory() + "/GitHub/mlx-examples/mnist/data", isDirectory: true)
if #available(macOS 13.0, *) {
    let fileURL = into.appending(component: "t10k-images-idx3-ubyte.gz")
    print("Path: " + fileURL.path())
    print("Exists: \(FileManager.default.fileExists(atPath: fileURL.path()))")
}
