import Foundation
print(FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!.path)
