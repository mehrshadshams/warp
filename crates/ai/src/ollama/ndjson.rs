//! Streaming NDJSON parser.
//!
//! Ollama returns one JSON object per line on `/api/chat?stream=true`. We
//! cannot rely on each chunk arriving aligned to a line boundary, so we
//! buffer until we see `\n`, parse, and emit. The parser is written as a
//! plain stateful struct so it can be unit-tested without spinning up an
//! HTTP client.

use serde::de::DeserializeOwned;

use super::error::OllamaError;

/// Caps the line buffer to protect against pathological / malicious payloads
/// (e.g. an upstream that never emits a newline). 1 MiB is well above any
/// realistic Ollama chunk.
const MAX_LINE_BYTES: usize = 1024 * 1024;

#[derive(Default)]
pub struct NdjsonParser {
    buf: Vec<u8>,
}

impl NdjsonParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes from the network. Returns the JSON-decoded
    /// objects that became complete with this chunk, in order.
    pub fn push<T: DeserializeOwned>(&mut self, bytes: &[u8]) -> Result<Vec<T>, OllamaError> {
        if self.buf.len().saturating_add(bytes.len()) > MAX_LINE_BYTES
            && !contains_newline(bytes)
            && !contains_newline(&self.buf)
        {
            return Err(OllamaError::Stream(format!(
                "Ollama NDJSON line exceeded {MAX_LINE_BYTES} bytes without a newline"
            )));
        }
        self.buf.extend_from_slice(bytes);

        let mut out = Vec::new();
        while let Some(idx) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=idx).collect();
            // Strip the trailing `\n` (and any `\r`).
            let trimmed = trim_line(&line);
            if trimmed.is_empty() {
                continue;
            }
            let value: T = serde_json::from_slice(trimmed)
                .map_err(|e| OllamaError::Protocol(format!("malformed NDJSON line: {e}")))?;
            out.push(value);
        }
        Ok(out)
    }

    /// Drains anything buffered that did not end with a newline. Ollama
    /// always terminates the final record with `\n`, but this is a safety
    /// net for proxies that strip the trailing newline.
    pub fn finish<T: DeserializeOwned>(&mut self) -> Result<Option<T>, OllamaError> {
        let trimmed = trim_line(&self.buf);
        if trimmed.is_empty() {
            self.buf.clear();
            return Ok(None);
        }
        let value: T = serde_json::from_slice(trimmed)
            .map_err(|e| OllamaError::Protocol(format!("malformed final NDJSON line: {e}")))?;
        self.buf.clear();
        Ok(Some(value))
    }
}

fn trim_line(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    while end > 0 && matches!(line[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    let mut start = 0;
    while start < end && line[start].is_ascii_whitespace() {
        start += 1;
    }
    &line[start..end]
}

fn contains_newline(b: &[u8]) -> bool {
    b.iter().any(|c| *c == b'\n')
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn emits_one_per_line() {
        let mut p = NdjsonParser::new();
        let out: Vec<Value> = p.push(b"{\"a\":1}\n{\"b\":2}\n").unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["a"], 1);
        assert_eq!(out[1]["b"], 2);
    }

    #[test]
    fn buffers_partial_lines_across_pushes() {
        let mut p = NdjsonParser::new();
        let out: Vec<Value> = p.push(b"{\"a\":").unwrap();
        assert!(out.is_empty());
        let out: Vec<Value> = p.push(b"1}\n").unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["a"], 1);
    }

    #[test]
    fn skips_blank_lines_and_handles_crlf() {
        let mut p = NdjsonParser::new();
        let out: Vec<Value> = p.push(b"\r\n{\"a\":1}\r\n\r\n{\"b\":2}\n").unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn finish_returns_remaining_unterminated_line() {
        let mut p = NdjsonParser::new();
        let _: Vec<Value> = p.push(b"{\"a\":1}").unwrap();
        let last: Option<Value> = p.finish().unwrap();
        assert_eq!(last.unwrap()["a"], 1);
    }

    #[test]
    fn rejects_garbage() {
        let mut p = NdjsonParser::new();
        let res: Result<Vec<Value>, _> = p.push(b"not json\n");
        assert!(matches!(res, Err(OllamaError::Protocol(_))));
    }
}
