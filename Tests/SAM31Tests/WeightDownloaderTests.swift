import Foundation
import Testing

@testable import SAM31

@Suite struct WeightDownloaderTests {
    @Test func buildsResolveURLs() {
        let d = WeightDownloader()
        #expect(
            d.url(for: "config.json").absoluteString
                == "https://huggingface.co/mlx-community/sam3.1-bf16/resolve/main/config.json")
        #expect(WeightDownloader.requiredFiles == ["config.json", "model.safetensors"])
    }

    @Test func buildsURLsForOtherReposAndRevisions() {
        let d = WeightDownloader(repo: "someone/sam3.1-fp16", revision: "a1b2c3")
        #expect(
            d.url(for: "model.safetensors").absoluteString
                == "https://huggingface.co/someone/sam3.1-fp16/resolve/a1b2c3/model.safetensors")
    }

    @Test func sendsTokenOnlyWhenSet() {
        let d = WeightDownloader()
        let anonymous = d.request(for: "config.json", method: "HEAD", environment: [:])
        #expect(anonymous.value(forHTTPHeaderField: "Authorization") == nil)
        #expect(anonymous.httpMethod == "HEAD")

        let empty = d.request(for: "config.json", method: "GET", environment: ["HF_TOKEN": ""])
        #expect(empty.value(forHTTPHeaderField: "Authorization") == nil)

        let authed = d.request(for: "config.json", method: "GET", environment: ["HF_TOKEN": "hf_test"])
        #expect(authed.value(forHTTPHeaderField: "Authorization") == "Bearer hf_test")
    }

    @Test func requestsTheRemainderWhenResuming() {
        let d = WeightDownloader()
        let r = d.request(for: "model.safetensors", method: "GET", environment: [:], resumeFrom: 1024)
        #expect(r.value(forHTTPHeaderField: "Range") == "bytes=1024-")
        let fresh = d.request(for: "model.safetensors", method: "GET", environment: [:])
        #expect(fresh.value(forHTTPHeaderField: "Range") == nil)
    }

    @Test func skipsOnlyCompleteFiles() {
        typealias A = WeightDownloader.Action
        // Complete file with a matching Content-Length: skip.
        #expect(WeightDownloader.action(existingSize: 100, partialSize: nil, expectedSize: 100) == A.skip)
        // Wrong size or unknown remote size: download again.
        #expect(WeightDownloader.action(existingSize: 99, partialSize: nil, expectedSize: 100) == A.fresh)
        #expect(WeightDownloader.action(existingSize: 100, partialSize: nil, expectedSize: nil) == A.fresh)
        // No file yet.
        #expect(WeightDownloader.action(existingSize: nil, partialSize: nil, expectedSize: 100) == A.fresh)
        // A shorter .partial resumes; an empty, oversized, or unverifiable one restarts.
        #expect(
            WeightDownloader.action(existingSize: nil, partialSize: 40, expectedSize: 100) == A.resume(40))
        #expect(WeightDownloader.action(existingSize: 99, partialSize: 40, expectedSize: 100) == A.resume(40))
        #expect(WeightDownloader.action(existingSize: nil, partialSize: 0, expectedSize: 100) == A.fresh)
        #expect(WeightDownloader.action(existingSize: nil, partialSize: 100, expectedSize: 100) == A.fresh)
        #expect(WeightDownloader.action(existingSize: nil, partialSize: 40, expectedSize: nil) == A.fresh)
    }

    @Test func readsSizesFromDisk() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam31-dl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("config.json")
        #expect(WeightDownloader.fileSize(at: file) == nil)
        try Data(repeating: 7, count: 123).write(to: file)
        #expect(WeightDownloader.fileSize(at: file) == 123)
    }

    @Test func redirectKeepsTokenOnlyForTheHub() {
        func request(_ url: String) -> URLRequest {
            var r = URLRequest(url: URL(string: url)!)
            r.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
            r.setValue("bytes=10-", forHTTPHeaderField: "Range")
            return r
        }
        let hub = WeightDownloader.redirectRequest(
            request("https://huggingface.co/api/resolve-cache/models/x/y/config.json"))
        #expect(hub.value(forHTTPHeaderField: "Authorization") == "Bearer secret")

        for url in [
            "https://cas-bridge.xethub.hf.co/file", "https://cdn-lfs.huggingface.co/file",
            "https://huggingface.co.evil.com/file",
        ] {
            let cdn = WeightDownloader.redirectRequest(request(url))
            #expect(cdn.value(forHTTPHeaderField: "Authorization") == nil, "\(url)")
            #expect(cdn.value(forHTTPHeaderField: "Range") == "bytes=10-")
            #expect(cdn.url == URL(string: url))
        }
    }
}
