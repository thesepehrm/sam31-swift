import Foundation

/// Downloads the SAM 3.1 checkpoint (`config.json` and `model.safetensors`) from the Hugging Face Hub.
///
/// Files that already exist with the size the server reports are skipped. Each file is written to
/// `<file>.partial` and moved into place only when complete, so an interrupted download resumes from
/// where it stopped on the next call. Cancelling the calling task stops the transfer.
///
/// The request carries `Authorization: Bearer <token>` only when the `HF_TOKEN` environment variable is
/// set. The default repository is public and needs no token.
///
/// The weights are released by Meta under the SAM License; downloading them means accepting it.
public struct WeightDownloader: Sendable {
    /// The files a weights directory needs, in download order.
    public static let requiredFiles = ["config.json", "model.safetensors"]

    /// Hugging Face repository, as `owner/name`.
    public let repo: String
    /// Branch, tag, or commit to download.
    public let revision: String
    private let session: URLSession

    private static let hub = URL(string: "https://huggingface.co")!

    /// Creates a downloader for `repo` at `revision`.
    public init(
        repo: String = "mlx-community/sam3.1-bf16", revision: String = "main", session: URLSession = .shared
    ) {
        self.repo = repo
        self.revision = revision
        self.session = session
    }

    /// The Hub URL of `file`: `https://huggingface.co/<repo>/resolve/<revision>/<file>`.
    public func url(for file: String) -> URL {
        Self.hub.appending(path: "\(repo)/resolve/\(revision)/\(file)")
    }

    /// Downloads every file in ``requiredFiles`` into `directory`, creating it if needed.
    ///
    /// - Parameter progress: called with the overall fraction done (0...1) as bytes arrive.
    /// - Throws: ``SAM31Error/download(_:)`` on a network, HTTP, or size error, or `CancellationError`
    ///   when the calling task is cancelled. A cancelled or failed download keeps its `.partial` file.
    public func download(to directory: URL, progress: (@Sendable (Double) -> Void)? = nil) async throws {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw SAM31Error.download("cannot create \(directory.path): \(error.localizedDescription)")
        }
        let environment = ProcessInfo.processInfo.environment
        var expected: [Int64?] = []
        for file in Self.requiredFiles {
            try Task.checkCancellation()
            expected.append(try await remoteSize(of: file, environment: environment))
        }
        let total = expected.reduce(Int64(0)) { $0 + ($1 ?? 0) }
        let report = DownloadProgress(total: total, callback: progress)
        report.update(0)

        for (file, size) in zip(Self.requiredFiles, expected) {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(file)
            let partial = directory.appendingPathComponent(file + ".partial")
            let action = Self.action(
                existingSize: Self.fileSize(at: destination), partialSize: Self.fileSize(at: partial),
                expectedSize: size)
            switch action {
            case .skip:
                report.finish(size ?? 0)
                continue
            case .fresh:
                try? FileManager.default.removeItem(at: partial)
                try await fetch(file, into: partial, offset: 0, environment: environment, report: report)
            case .resume(let offset):
                try await fetch(file, into: partial, offset: offset, environment: environment, report: report)
            }
            let got = Self.fileSize(at: partial) ?? 0
            if let size, got != size {
                throw SAM31Error.download("\(file): received \(got) bytes, expected \(size)")
            }
            do {
                _ = try FileManager.default.replaceItemAt(destination, withItemAt: partial)
            } catch {
                throw SAM31Error.download("cannot move \(file) into place: \(error.localizedDescription)")
            }
            report.finish(got)
        }
        progress?(1)
    }

    // MARK: - Pure helpers (unit tested)

    /// What to do with one file.
    enum Action: Equatable {
        /// The file is complete.
        case skip
        /// Download the whole file.
        case fresh
        /// Append the rest of the file to a `.partial` of this many bytes.
        case resume(Int64)
    }

    /// Chooses the action for a file from the sizes on disk and the size the server reports.
    /// Only a size match counts as complete; without a remote size nothing can be verified or resumed.
    static func action(existingSize: Int64?, partialSize: Int64?, expectedSize: Int64?) -> Action {
        guard let expectedSize else { return .fresh }
        if existingSize == expectedSize { return .skip }
        if let partialSize, partialSize > 0, partialSize < expectedSize { return .resume(partialSize) }
        return .fresh
    }

    /// The size of the regular file at `url`, or nil if there is none.
    static func fileSize(at url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
            attributes[.type] as? FileAttributeType == .typeRegular,
            let size = attributes[.size] as? NSNumber
        else { return nil }
        return size.int64Value
    }

    /// A request for `file`. It asks for the raw bytes (no transfer compression), so sizes match the
    /// file on disk, and adds a bearer token only when `HF_TOKEN` is set and non-empty.
    func request(
        for file: String, method: String, environment: [String: String], resumeFrom offset: Int64? = nil
    ) -> URLRequest {
        var request = URLRequest(url: url(for: file))
        request.httpMethod = method
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        if let token = environment["HF_TOKEN"], !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let offset {
            request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
        }
        return request
    }

    // MARK: - Network

    /// The file size the Hub reports: `X-Linked-Size` for LFS files, else the final `Content-Length`.
    private func remoteSize(of file: String, environment: [String: String]) async throws -> Int64? {
        let request = request(for: file, method: "HEAD", environment: environment)
        let response: URLResponse
        do {
            (_, response) = try await session.data(for: request)
        } catch {
            throw Self.networkError(error, file: file)
        }
        let http = try Self.checked(response, file: file, allowed: [200])
        if let linked = http.value(forHTTPHeaderField: "X-Linked-Size"), let size = Int64(linked) {
            return size
        }
        return http.expectedContentLength >= 0 ? http.expectedContentLength : nil
    }

    /// Downloads `file` (from `offset` on), streaming it into `partial` as bytes arrive, so a
    /// cancelled or failed transfer keeps what it received.
    private func fetch(
        _ file: String, into partial: URL, offset: Int64, environment: [String: String],
        report: DownloadProgress
    ) async throws {
        let request = request(
            for: file, method: "GET", environment: environment, resumeFrom: offset > 0 ? offset : nil)
        if offset == 0 {
            guard FileManager.default.createFile(atPath: partial.path, contents: nil) else {
                throw SAM31Error.download("cannot create \(partial.path)")
            }
        }
        let writer: PartialFileWriter
        do {
            writer = try PartialFileWriter(file: file, url: partial, offset: offset, report: report)
        } catch {
            throw SAM31Error.download("cannot open \(partial.path): \(error.localizedDescription)")
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                writer.start(session.dataTask(with: request), continuation: continuation)
            }
        } onCancel: {
            writer.cancel()
        }
    }

    private static func checked(_ response: URLResponse, file: String, allowed: Set<Int>) throws
        -> HTTPURLResponse
    {
        guard let http = response as? HTTPURLResponse else {
            throw SAM31Error.download("\(file): not an HTTP response")
        }
        guard allowed.contains(http.statusCode) else {
            let hint =
                [401, 403].contains(http.statusCode) ? " (set HF_TOKEN for a gated or private repo)" : ""
            throw SAM31Error.download("\(file): HTTP \(http.statusCode)\(hint)")
        }
        return http
    }

    private static func networkError(_ error: Error, file: String) -> Error {
        if error is CancellationError { return error }
        if let url = error as? URLError, url.code == .cancelled { return CancellationError() }
        return SAM31Error.download("\(file): \(error.localizedDescription)")
    }
}

/// Streams one HTTP response body into a `.partial` file: appends after a 206 (range) response,
/// truncates after a 200 (the whole file). Receives the task's delegate callbacks; a cancel request
/// may arrive before the task exists.
private final class PartialFileWriter: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let file: String
    private let handle: FileHandle
    private let offset: Int64
    private let report: DownloadProgress
    private var task: URLSessionDataTask?
    private var continuation: CheckedContinuation<Void, Error>?
    private var cancelled = false
    private var failure: Error?
    private var written: Int64 = 0

    init(file: String, url: URL, offset: Int64, report: DownloadProgress) throws {
        self.file = file
        self.handle = try FileHandle(forWritingTo: url)
        self.offset = offset
        self.report = report
    }

    func start(_ task: URLSessionDataTask, continuation: CheckedContinuation<Void, Error>) {
        task.delegate = self
        lock.lock()
        self.task = task
        self.continuation = continuation
        let cancelled = self.cancelled
        lock.unlock()
        if cancelled { task.cancel() }
        task.resume()
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let task = self.task
        lock.unlock()
        task?.cancel()
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        do {
            guard let http = response as? HTTPURLResponse else {
                throw SAM31Error.download("\(file): not an HTTP response")
            }
            switch http.statusCode {
            case 206 where offset > 0:
                try handle.seekToEnd()
                written = offset
            case 200:
                // The whole file, even when a range was asked for.
                try handle.truncate(atOffset: 0)
                written = 0
            default:
                let hint =
                    [401, 403].contains(http.statusCode) ? " (set HF_TOKEN for a gated or private repo)" : ""
                throw SAM31Error.download("\(file): HTTP \(http.statusCode)\(hint)")
            }
            completionHandler(.allow)
        } catch {
            fail(error)
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try handle.write(contentsOf: data)
            written += Int64(data.count)
            report.update(written)
        } catch {
            fail(SAM31Error.download("cannot write \(file): \(error.localizedDescription)"))
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle.synchronize()
        try? handle.close()
        lock.lock()
        let continuation = self.continuation
        self.continuation = nil
        let failure = self.failure
        let cancelled = self.cancelled
        lock.unlock()
        if let failure {
            continuation?.resume(throwing: failure)
        } else if cancelled || (error as? URLError)?.code == .cancelled {
            continuation?.resume(throwing: CancellationError())
        } else if let error {
            continuation?.resume(throwing: SAM31Error.download("\(file): \(error.localizedDescription)"))
        } else {
            continuation?.resume()
        }
    }

    private func fail(_ error: Error) {
        lock.lock()
        if failure == nil { failure = error }
        lock.unlock()
    }
}

/// Overall progress across files: bytes of finished files plus the file in flight.
private final class DownloadProgress: @unchecked Sendable {
    private let lock = NSLock()
    private let total: Int64
    private let callback: (@Sendable (Double) -> Void)?
    private var finished: Int64 = 0

    init(total: Int64, callback: (@Sendable (Double) -> Void)?) {
        self.total = total
        self.callback = callback
    }

    func update(_ inFlight: Int64) {
        guard let callback, total > 0 else { return }
        lock.lock()
        let done = finished + inFlight
        lock.unlock()
        callback(min(1, Double(done) / Double(total)))
    }

    func finish(_ bytes: Int64) {
        lock.lock()
        finished += bytes
        lock.unlock()
        update(0)
    }
}
