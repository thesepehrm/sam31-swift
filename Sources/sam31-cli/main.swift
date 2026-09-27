import Foundation
import SAM31

/// Runs one command. Returns the process exit status: 0 on success, 1 on failure, 64 on bad usage.
func run(_ arguments: [String]) async -> Int32 {
    guard let command = arguments.first, !["-h", "--help", "help"].contains(command) else {
        print(usage)
        return arguments.isEmpty ? 64 : 0
    }
    let rest = arguments.dropFirst()
    if rest.contains("-h") || rest.contains("--help") {
        print(usage)
        return 0
    }
    let common: Set<String> = ["weights", "dtype"]
    do {
        switch command {
        case "download":
            try await runDownload(Options(rest, allowed: ["dir", "repo", "revision"]))
        case "segment":
            try await runSegment(Options(rest, allowed: common.union(["image", "point", "box", "out"])))
        case "detect":
            try await runDetect(
                Options(rest, allowed: common.union(["image", "text", "threshold", "out-dir"])))
        case "track":
            try await runTrack(Options(rest, allowed: common.union(["video", "point", "frames", "out-dir"])))
        case "bench":
            try await runBench(Options(rest, allowed: common.union(["video", "frames", "out-dir"])))
        default:
            throw UsageError("unknown command '\(command)'")
        }
        return 0
    } catch let error as UsageError {
        printError("error: \(error.message)\n\n\(usage)")
        return 64
    } catch let error as CLIError {
        printError("error: \(error.message)")
    } catch let error as SAM31Error {
        printError("error: \(describe(error))")
    } catch is CancellationError {
        printError("error: cancelled")
    } catch {
        printError("error: \(error.localizedDescription)")
    }
    return 1
}

func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

exit(await run(Array(CommandLine.arguments.dropFirst())))
