import Foundation
import Testing

@testable import SAM31

/// Every entry of the fixture corpus (parity/tokenizer_corpus.txt run through HF CLIPTokenizer).
@Suite(.enabled(if: Fixtures.available), .serialized) struct TokenizerParityTests {
    @Test func fullCorpusMatchesHF() throws {
        let data = try Data(contentsOf: Fixtures.fixturesURL!.appending(path: "tokenizer.json"))
        let cases = try JSONDecoder().decode([TokenCase].self, from: data)
        #expect(cases.count == Fixtures.manifest["tokenizer_entries"] as? Int)
        let tok = try CLIPTokenizer.bundled()
        var mismatches = 0
        for c in cases {
            let e = tok.encode(c.text)
            if e.inputIds != c.inputIds || e.attentionMask != c.attentionMask { mismatches += 1 }
            #expect(e.inputIds == c.inputIds, "text: \(c.text.debugDescription)")
            #expect(e.attentionMask == c.attentionMask, "text: \(c.text.debugDescription)")
        }
        #expect(mismatches == 0, "\(mismatches)/\(cases.count) corpus entries differ")
    }
}
