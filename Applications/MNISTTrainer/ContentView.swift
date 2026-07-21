// Copyright © 2024 Apple Inc.

import MLX
import MLXMNIST
import MLXNN
import MLXOptimizers
import SwiftUI

struct TrainingView: View {

    @Binding var trainer: ModelState

    var body: some View {
        VStack {
            Spacer()

            ScrollView(.vertical) {
                ForEach(trainer.messages, id: \.self) {
                    Text($0)
                }
            }

            HStack {
                Spacer()
                switch trainer.state {
                case .untrained:
                    Button("Train") {
                        Task {
                            do {
                                try await trainer.train()
                            } catch {
                                trainer.messages.append("Training failed: \(error.localizedDescription)")
                            }
                        }
                    }
                case .trained(let model), .predict(let model):
                    Button("Draw a digit") {
                        trainer.state = .predict(model)
                    }
                }

                Spacer()
            }
            Spacer()
        }
        .padding()
    }
}

struct ContentView: View {
    // the training loop
    @State var trainer = ModelState()

    var body: some View {
        switch trainer.state {
        case .untrained, .trained:
            TrainingView(trainer: $trainer)
        case .predict(let model):
            PredictionView(model: model)
        }
    }
}

@MainActor
@Observable
class ModelState {

    enum State {
        case untrained
        case trained(LeNetContainer)
        case predict(LeNetContainer)
    }

    var state: State = .untrained
    var messages = [String]()

    func train() async throws {
        #if targetEnvironment(simulator)
            messages.append(
                "MLX cannot evaluate on the iOS Simulator. Run this app on a real iPhone/iPad or use Mac (Designed for iPad)."
            )
            return
        #endif

        let model = LeNetContainer()
        if try await model.loadIfAvailable() {
            messages.append("Loaded saved model weights.")
            self.state = .trained(model)
            return
        }

        try await model.train(output: self)
        self.state = .trained(model)
    }
}

actor LeNetContainer {

    private var model: LeNet?

    let mnistImageSize: CGSize = CGSize(width: 28, height: 28)

    private var modelURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MNISTTrainer/Models/lenet.safetensors")
    }

    func loadIfAvailable() throws -> Bool {
        #if targetEnvironment(simulator)
            let device = Device(.cpu)
        #else
            let device = Device(.gpu)
        #endif

        return try Device.withDefaultDevice(device) {
            guard FileManager.default.fileExists(atPath: modelURL.path) else { return false }

            let loadedWeights = try loadArrays(url: modelURL)
            let model = LeNet()
            try model.update(
                parameters: ModuleParameters.unflattened(Array(loadedWeights)), verify: .none)
            eval(model.parameters())
            self.model = model
            return true
        }
    }

    func train(output: ModelState) async throws {
        #if targetEnvironment(simulator)
            let device = Device(.cpu)
        #else
            let device = Device(.gpu)
        #endif

        try await Device.withDefaultDevice(device) {
            try await trainWithDefaultDevice(output: output)
        }
    }

    private func trainWithDefaultDevice(output: ModelState) async throws {
        let model = LeNet()
        self.model = model

        // Note: this is pretty close to the code in `mnist-tool`, just
        // wrapped in an Observable to make it easy to display in SwiftUI

        // download & load the training data
        let fileManager = FileManager.default
        let url = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("mnist/data", isDirectory: true)
        let legacyURL = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("mnist/data", isDirectory: true)

        // Preserve data downloaded by older versions that stored it in Caches.
        if !fileManager.fileExists(atPath: url.path), fileManager.fileExists(atPath: legacyURL.path) {
            try fileManager.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fileManager.moveItem(at: legacyURL, to: url)
        }

        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        try await download(into: url)
        let data = try load(from: url)

        let trainImages = data[.init(.training, .images)]!
        let trainLabels = data[.init(.training, .labels)]!
        let testImages = data[.init(.test, .images)]!
        let testLabels = data[.init(.test, .labels)]!

        eval(model.parameters())

        // the training loop
        let lg = valueAndGrad(model: model, loss)
        let optimizer = SGD(learningRate: 0.1)

        // using a consistent random seed so it behaves the same way each time
        MLXRandom.seed(0)
        var generator: RandomNumberGenerator = SplitMix64(seed: 0)

        for e in 0 ..< 10 {
            let start = Date.timeIntervalSinceReferenceDate

            for (x, y) in iterateBatches(
                batchSize: 256, x: trainImages, y: trainLabels, using: &generator)
            {
                // loss and gradients
                let (_, grads) = lg(model, x, y)

                // use SGD to update the weights
                optimizer.update(model: model, gradients: grads)

                // eval the parameters so the next iteration is independent
                eval(model, optimizer)
            }

            let accuracy = eval(model: model, x: testImages, y: testLabels)

            let end = Date.timeIntervalSinceReferenceDate

            // add to messages -- triggers display
            let accuracyItem = accuracy.item(Float.self)
            await MainActor.run {
                output.messages.append(
                    """
                    Epoch \(e): test accuracy \(accuracyItem.formatted())
                    Time: \((end - start).formatted())

                    """
                )
            }
        }

        let saveFileManager = FileManager.default
        try saveFileManager.createDirectory(
            at: modelURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: weights, url: modelURL)
        let savedModelName = modelURL.lastPathComponent

        await MainActor.run {
            output.messages.append("Saved model weights to \(savedModelName).")
        }
    }

    func evaluate(image: CGImage) -> Int? {
        guard let model else { return nil }

        let pixelData = image.grayscaleImage(with: mnistImageSize)?.pixelData()
        if let pixelData {
            let x = pixelData.reshaped([1, 28, 28, 1]).asType(.float32) / 255.0
            return argMax(model(x)).item()
        } else {
            return nil
        }
    }
}
