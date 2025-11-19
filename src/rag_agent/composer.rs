use crate::rag_agent::{RAGFrame, RetrievedSegment};

pub fn compose_answer(query: &str, frame: &RAGFrame) -> String {
    if frame.combined.is_empty() {
        return "I'm not sure yet—tell me more.".into();
    }

    let mut out = String::new();
    out.push_str("Here’s what I know:\n");

    for seg in frame.combined.iter().take(3) {
        out.push_str("- ");
        out.push_str(&seg.text);
        out.push('\n');
    }

    out
}
