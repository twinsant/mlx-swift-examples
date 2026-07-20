import Foundation

let into = URL(fileURLWithPath: NSHomeDirectory() + "/GitHub/mlx-examples/mnist/data", isDirectory: true)
let fileURL = into.appendingPathComponent("t10k-images-idx3-ubyte.gz")
print("Path1: " + fileURL.path)
if #available(macOS 13.0, *) {
    print("Path2: " + fileURL.path())
}

let exists1 = FileManager.default.fileExists(atPath: fileURL.path)
if #available(macOS 13.0, *) {
    let exists2 = FileManager.default.fileExists(atPath: fileURL.path())
    print("Exists2: \(exists2)")
}
print("Exists1: \(exists1)")

