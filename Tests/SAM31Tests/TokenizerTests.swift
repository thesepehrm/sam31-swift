import Foundation
import Testing

@testable import SAM31

struct TokenCase: Decodable, Sendable {
    let text: String
    let inputIds: [Int32]
    let attentionMask: [Int32]

    enum CodingKeys: String, CodingKey {
        case text
        case inputIds = "input_ids"
        case attentionMask = "attention_mask"
    }
}

extension TokenCase: CustomTestStringConvertible {
    var testDescription: String { text.debugDescription }
}

@Suite struct TokenizerTests {
    let tok = try! CLIPTokenizer.bundled()

    @Test func knownPrompt() {
        let e = tok.encode("The Dancer!")
        #expect(Array(e.inputIds.prefix(5)) == [49406, 518, 11400, 256, 49407])
        #expect(e.inputIds.count == 32 && e.attentionMask.prefix(6) == [1, 1, 1, 1, 1, 0])
    }

    /// Values probed from HF `CLIPTokenizer` (transformers 5.17, tokenizers backend).
    @Test(arguments: [
        // Lowercase is per scalar: no final-sigma rule, and İ expands to i + U+0307.
        ("ΣΑΣ", [49406, 139, 225, 138, 109, 139, 481, 49407] as [Int32]),
        ("İstanbul", [49406, 328, 16384, 11231, 49407]),
        // Raw `<|endoftext|>` is a special token; `<|startoftext|>` matches after normalization.
        ("hello<|endoftext|>world", [49406, 3306, 49407, 1002, 49407]),
        ("A<|startoftext|>b", [49406, 320, 49406, 321, 49407]),
        ("<|STARTOFTEXT|>", [49406, 49406, 49407]),
        // Upper-case `<|ENDOFTEXT|>` is not special: it is lowercased and split as `<|`, letters, `|>`.
        ("x <|ENDOFTEXT|>", [49406, 343, 27, 347, 40786, 4160, 91, 285, 49407]),
    ])
    func hfEdgeCases(_ text: String, _ expected: [Int32]) {
        let e = tok.encode(text)
        #expect(Array(e.inputIds.prefix(expected.count)) == expected)
        #expect(e.attentionMask.reduce(0, +) == Int32(expected.count))
    }

    @Test func truncatesAndPads() {
        let e = tok.encode(String(repeating: "cat ", count: 100), maxLength: 8)
        #expect(e.inputIds == [49406] + Array(repeating: 2368, count: 6) + [49407])
        #expect(e.attentionMask == Array(repeating: 1, count: 8))
        let empty = tok.encode("")
        #expect(empty.inputIds == [49406] + Array(repeating: 49407, count: 31))
        #expect(empty.attentionMask == [1, 1] + Array(repeating: 0, count: 30))
    }

    static let golden: [TokenCase] = {
        let url = Bundle.module.url(forResource: "Resources/tokenizer-golden", withExtension: "json")!
        return try! JSONDecoder().decode([TokenCase].self, from: Data(contentsOf: url))
    }()

    @Test(arguments: golden) func matchesHF(_ c: TokenCase) {
        let e = tok.encode(c.text)
        #expect(e.inputIds == c.inputIds, "text: \(c.text.debugDescription)")
        #expect(e.attentionMask == c.attentionMask)
    }
}
