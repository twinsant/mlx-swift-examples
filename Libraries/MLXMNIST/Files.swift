// Copyright © 2024 Apple Inc.

import Foundation
import Gzip
import MLX

// based on https://github.com/ml-explore/mlx-examples/blob/main/mnist/mnist.py

public enum Use: String, Hashable, Sendable {
    case test
    case training
}

public enum DataKind: String, Hashable, Sendable {
    case images
    case labels
}

public struct FileKind: Hashable, CustomStringConvertible, Sendable {
    let use: Use
    let data: DataKind

    public init(_ use: Use, _ data: DataKind) {
        self.use = use
        self.data = data
    }

    public var description: String {
        "\(use.rawValue)-\(data.rawValue)"
    }
}

struct LoadInfo: Sendable {
    let name: String
    let offset: Int
    let convert: @Sendable (MLXArray) -> MLXArray
}

private let baseURLs = [
    URL(string: "https://storage.googleapis.com/cvdf-datasets/mnist/")!,
    URL(string: "https://ossci-datasets.s3.amazonaws.com/mnist/")!,
    URL(string: "https://aiml.4everbucket.com/mnist/")!,
]

private struct DownloadError: LocalizedError {
    let fileName: String
    let lastError: Error?

    var errorDescription: String? {
        if let lastError {
            return "Unable to download \(fileName) after trying all MNIST mirrors: \(lastError.localizedDescription)"
        }
        return "Unable to download \(fileName) after trying all MNIST mirrors."
    }
}

private let files = [
    FileKind(.training, .images): LoadInfo(
        name: "train-images-idx3-ubyte.gz",
        offset: 16,
        convert: {
            $0.reshaped([-1, 28, 28, 1]).asType(.float32) / 255.0
        }),
    FileKind(.test, .images): LoadInfo(
        name: "t10k-images-idx3-ubyte.gz",
        offset: 16,
        convert: {
            $0.reshaped([-1, 28, 28, 1]).asType(.float32) / 255.0
        }),
    FileKind(.training, .labels): LoadInfo(
        name: "train-labels-idx1-ubyte.gz",
        offset: 8,
        convert: {
            $0.asType(.uint32)
        }),
    FileKind(.test, .labels): LoadInfo(
        name: "t10k-labels-idx1-ubyte.gz",
        offset: 8,
        convert: {
            $0.asType(.uint32)
        }),
]

public func download(into: URL) async throws {
    for (_, info) in files {
        let fileURL = into.appending(component: info.name)
        if !FileManager.default.fileExists(atPath: fileURL.path()) {
            print("Download: \(info.name)")
            var lastError: Error?
            var downloaded = false

            for baseURL in baseURLs {
                let url = baseURL.appending(component: info.name)
                var request = URLRequest(url: url)
                request.timeoutInterval = 60

                for attempt in 0 ..< 3 {
                    do {
                        let configuration = URLSessionConfiguration.ephemeral
                        configuration.timeoutIntervalForRequest = 60
                        configuration.timeoutIntervalForResource = 300
                        let session = URLSession(configuration: configuration)
                        defer { session.invalidateAndCancel() }

                        let (data, response) = try await session.data(for: request)
                        guard let httpResponse = response as? HTTPURLResponse,
                              (200 ..< 300).contains(httpResponse.statusCode)
                        else {
                            throw URLError(.badServerResponse)
                        }

                        try data.write(to: fileURL, options: .atomic)
                        downloaded = true
                        break
                    } catch {
                        lastError = error
                        if attempt < 2 {
                            try await Task.sleep(nanoseconds: UInt64(attempt + 1) * 1_000_000_000)
                        }
                    }
                }

                if downloaded { break }
            }

            if !downloaded {
                throw DownloadError(fileName: info.name, lastError: lastError)
            }
        }
    }
}

public func load(from: URL) throws -> [FileKind: MLXArray] {
    var result = [FileKind: MLXArray]()

    for (key, info) in files {
        let fileURL = from.appending(component: info.name)
        let data = try Data(contentsOf: fileURL).gunzipped()

        let array = MLXArray(
            data.dropFirst(info.offset), [data.count - info.offset], type: UInt8.self)

        result[key] = info.convert(array)
    }

    return result
}
