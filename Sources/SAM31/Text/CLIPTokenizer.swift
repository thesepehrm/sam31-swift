// Port of transformers/models/clip/tokenization_clip.py::CLIPTokenizer (transformers 5.17, tokenizers
// backend), as loaded by mlx_vlm/models/sam3/processing_sam3.py for openai/clip-vit-base-patch32.
//
// Pipeline, matching the HF `tokenizers` graph exactly:
//   1. Added tokens: raw text is split on `<|endoftext|>` (normalized=False) before normalization;
//      `<|startoftext|>` (normalized=True) is split out after normalization.
//   2. Normalizer: NFC, then every run of Unicode White_Space becomes one " ", then per-scalar
//      lowercase (full mapping, no final-sigma context rule).
//   3. Pre-tokenizer: the CLIP split regex, then ByteLevel (GPT-2 regex re-split + byte-to-unicode).
//      Both regexes are folded into one hand-rolled scanner; see `preTokenize`.
//   4. BPE with `</w>` appended to each word's last symbol, merges ranked by merges.txt order.
//   5. `[bos] + ids[..<maxLength-2] + [eos]`, right-padded with `pad`.
import Foundation

enum CLIPTokenizerError: Error {
    case missingResource(String)
    case malformedResource(String)
}

struct CLIPTokenizer: Sendable {
    static let bos: Int32 = 49406
    static let eos: Int32 = 49407
    static let pad: Int32 = 49407

    private static let startOfText = Array("<|startoftext|>".unicodeScalars)
    private static let endOfText = Array("<|endoftext|>".unicodeScalars)

    private struct Pair: Hashable, Sendable {
        let first: String
        let second: String
    }

    private let encoder: [String: Int32]
    private let ranks: [Pair: Int]
    private let byteEncoder: [String]
    private let cache = BPECache()

    init(vocab: [String: Int32], merges: [(String, String)]) {
        encoder = vocab
        var ranks: [Pair: Int] = [:]
        for (i, m) in merges.enumerated() where ranks[Pair(first: m.0, second: m.1)] == nil {
            ranks[Pair(first: m.0, second: m.1)] = i
        }
        self.ranks = ranks
        byteEncoder = Self.bytesToUnicode()
    }

    private static let shared = Result { try load() }

    /// The tokenizer built from the bundled `openai/clip-vit-base-patch32` vocab and merges.
    static func bundled() throws -> CLIPTokenizer { try shared.get() }

    private static func load() throws -> CLIPTokenizer {
        func url(_ name: String, _ ext: String) throws -> URL {
            guard let u = Bundle.module.url(forResource: "Resources/\(name)", withExtension: ext) else {
                throw CLIPTokenizerError.missingResource("\(name).\(ext)")
            }
            return u
        }
        let vocab = try JSONDecoder().decode(
            [String: Int32].self, from: Data(contentsOf: url("clip-vocab", "json")))
        let text = try String(contentsOf: url("clip-merges", "txt"), encoding: .utf8)
        var merges: [(String, String)] = []
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#version") { continue }
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { throw CLIPTokenizerError.malformedResource(String(line)) }
            merges.append((String(parts[0]), String(parts[1])))
        }
        return CLIPTokenizer(vocab: vocab, merges: merges)
    }

    /// Token IDs and attention mask, identical to HF
    /// `tokenizer(text, padding="max_length", max_length=maxLength, truncation=True)`.
    func encode(_ text: String, maxLength: Int = 32) -> (inputIds: [Int32], attentionMask: [Int32]) {
        precondition(maxLength >= 2, "maxLength must leave room for BOS and EOS")
        var ids: [Int32] = []
        for (i, raw) in Self.split(Array(text.unicodeScalars), on: Self.endOfText).enumerated() {
            if i > 0 { ids.append(Self.eos) }
            for (j, piece) in Self.split(normalize(raw), on: Self.startOfText).enumerated() {
                if j > 0 { ids.append(Self.bos) }
                for word in preTokenize(piece) { ids += bpe(word) }
            }
        }
        let body = [Self.bos] + ids.prefix(maxLength - 2) + [Self.eos]
        let padding = maxLength - body.count
        return (
            body + Array(repeating: Self.pad, count: padding),
            Array(repeating: 1, count: body.count) + Array(repeating: 0, count: padding)
        )
    }

    // MARK: - Added-token splitting

    /// `scalars` split on every occurrence of `token` (the token itself is dropped).
    private static func split(_ scalars: [Unicode.Scalar], on token: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var parts: [[Unicode.Scalar]] = []
        var start = 0
        var i = 0
        while i + token.count <= scalars.count {
            if scalars[i..<i + token.count].elementsEqual(token) {
                parts.append(Array(scalars[start..<i]))
                i += token.count
                start = i
            } else {
                i += 1
            }
        }
        parts.append(Array(scalars[start...]))
        return parts
    }

    // MARK: - Normalizer

    private func normalize(_ scalars: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var s = String.UnicodeScalarView()
        s.append(contentsOf: scalars)
        let nfc = String(s).precomposedStringWithCanonicalMapping
        var out: [Unicode.Scalar] = []
        out.reserveCapacity(nfc.unicodeScalars.count)
        var inWhitespace = false
        for c in nfc.unicodeScalars {
            if c.properties.isWhitespace {
                if !inWhitespace { out.append(" ") }
                inWhitespace = true
            } else {
                inWhitespace = false
                out.append(contentsOf: c.properties.lowercaseMapping.unicodeScalars)
            }
        }
        return out
    }

    // MARK: - Pre-tokenizer

    private static func isLetter(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: true
        default: false
        }
    }

    private static func isNumber(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: true
        default: false
        }
    }

    private static let contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"].map {
        Array($0.unicodeScalars)
    }

    /// Scanner equivalent of the CLIP split regex followed by ByteLevel's GPT-2 regex. The CLIP regex
    /// is `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`
    /// (first alternative wins at each position, whitespace is removed). The GPT-2 re-split is a no-op
    /// on every CLIP piece except the two special literals, which it breaks back into `<|`, letters,
    /// `|>`: exactly what the remaining alternatives produce, so the literals are left out here.
    /// Returns each piece's UTF-8 bytes.
    private func preTokenize(_ s: [Unicode.Scalar]) -> [[UInt8]] {
        func hasPrefix(_ p: [Unicode.Scalar], at i: Int) -> Bool {
            i + p.count <= s.count && s[i..<i + p.count].elementsEqual(p)
        }
        var words: [[UInt8]] = []
        var i = 0
        while i < s.count {
            let c = s[i]
            var end = i + 1
            if c.properties.isWhitespace {
                i += 1
                continue
            } else if let lit = Self.contractions.first(where: { hasPrefix($0, at: i) }) {
                end = i + lit.count
            } else if Self.isLetter(c) {
                while end < s.count && Self.isLetter(s[end]) { end += 1 }
            } else if !Self.isNumber(c) {
                while end < s.count, !s[end].properties.isWhitespace, !Self.isLetter(s[end]),
                    !Self.isNumber(s[end])
                {
                    end += 1
                }
            }
            var piece = String.UnicodeScalarView()
            piece.append(contentsOf: s[i..<end])
            words.append(Array(String(piece).utf8))
            i = end
        }
        return words
    }

    // MARK: - Byte-level BPE

    /// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest to U+0100 onwards.
    private static func bytesToUnicode() -> [String] {
        let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var table = [String](repeating: "", count: 256)
        var n = 0
        for b in 0..<256 {
            if printable.contains(b) {
                table[b] = String(Unicode.Scalar(UInt8(b)))
            } else {
                table[b] = String(Unicode.Scalar(UInt32(256 + n))!)
                n += 1
            }
        }
        return table
    }

    private func bpe(_ bytes: [UInt8]) -> [Int32] {
        if let hit = cache[bytes] { return hit }
        var symbols = bytes.map { byteEncoder[Int($0)] }
        symbols[symbols.count - 1] += "</w>"
        while symbols.count > 1 {
            var best: (rank: Int, pair: Pair)?
            for k in 0..<symbols.count - 1 {
                let p = Pair(first: symbols[k], second: symbols[k + 1])
                if let r = ranks[p], r < best?.rank ?? .max { best = (r, p) }
            }
            guard let pair = best?.pair else { break }
            var merged: [String] = []
            merged.reserveCapacity(symbols.count)
            var k = 0
            while k < symbols.count {
                if k + 1 < symbols.count && symbols[k] == pair.first && symbols[k + 1] == pair.second {
                    merged.append(pair.first + pair.second)
                    k += 2
                } else {
                    merged.append(symbols[k])
                    k += 1
                }
            }
            symbols = merged
        }
        // Every byte symbol is in the vocab, so the unk fallback (`<|endoftext|>`) never fires in practice.
        let ids = symbols.map { encoder[$0] ?? Self.eos }
        cache[bytes] = ids
        return ids
    }
}

/// Per-word BPE memo shared across copies of a tokenizer.
private final class BPECache: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [[UInt8]: [Int32]] = [:]

    subscript(key: [UInt8]) -> [Int32]? {
        get { lock.withLock { storage[key] } }
        set { lock.withLock { storage[key] = newValue } }
    }
}
