// PFC/cortex_solver.rs
use crate::PFC::cortex::{CortexInputFrame, CortexReasoningMode, CortexAction};
use crate::PFC::context_retriever::ContextFrame;

#[derive(Debug)]
pub struct CortexDraftOutput {
    pub response: Option<String>,
    pub action: Option<CortexAction>,
    pub trace: Option<String>,
}

/* ------------ tiny helpers (no deps) ------------ */

fn is_question(s: &str) -> bool {
    let s = s.trim().to_lowercase();
    s.ends_with('?')
        || s.starts_with("what ")
        || s.starts_with("who ")
        || s.starts_with("where ")
        || s.starts_with("when ")
        || s.starts_with("why ")
        || s.starts_with("how ")
        || s.starts_with("which ")
        || s.starts_with("what's ")
        || s.starts_with("whats ")
}

fn normalize_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

/// Extract a short “needle” from the query, e.g. “launch code” out of “what is the launch code?”
fn extract_needle(query: &str) -> Option<String> {
    let q = query.trim().to_lowercase();
    let mut s = q.as_str();

    // strip common prefixes
    for p in [
        "what is ", "what's ", "whats ", "what are ", "who is ", "who are ",
        "where is ", "where are ", "how do i ", "how to ", "how does ",
        "which ", "when is ", "when are ", "define ", "explain ",
    ] {
        if s.starts_with(p) {
            s = &s[p.len()..];
            break;
        }
    }

    // drop trailing '?'
    s = s.trim_end_matches('?').trim();

    // strip boring determiners
    for d in ["the ", "a ", "an ", "my ", "your "] {
        if s.starts_with(d) {
            s = &s[d.len()..];
        }
    }

    if s.is_empty() { None } else { Some(s.to_string()) }
}

/// Return the top sentence from `text` that best matches `needle` (very light heuristic).
// before:
// fn best_sentence_for<'a>(text: &'a str, needle: &str) -> Option<&'a str>

// after:
fn best_sentence_for(text: &str, needle: &str) -> Option<String> {
    let t = normalize_ws(text); // owns the cleaned text
    let mut best: Option<(&str, usize)> = None;

    for raw in t.split(|c| c == '.' || c == '!' || c == '?') {
        let sent = raw.trim();
        if sent.is_empty() { continue; }

        let score = if sent.to_lowercase().contains(needle) {
            1000
        } else {
            let nl: Vec<&str> = needle.split_whitespace().collect();
            nl.iter().filter(|tok| sent.to_lowercase().contains(*tok)).count()
        };

        match best {
            None => best = Some((sent, score)),
            Some((_, bs)) if score > bs => best = Some((sent, score)),
            _ => {}
        }
    }

    best.map(|(s, _)| s.to_string()) // return owned String
}

/// Summarize top internal hits for traces / debug
fn summarize_hits(ctx: &ContextFrame, max_show: usize) -> (String, usize) {
    let n = ctx.top_hits.len();
    let show = n.min(max_show);
    let mut lines = Vec::with_capacity(show + 2);
    lines.push(format!("Found {n} related memories; showing top {show}:"));
    for (i, hit) in ctx.top_hits.iter().take(show).enumerate() {
        let snippet = if hit.text.len() > 160 {
            format!("{}…", &hit.text[..160])
        } else {
            hit.text.clone()
        };
        lines.push(format!("{}. [sim {:.2}] {}", i + 1, hit.sim, snippet));
    }
    (lines.join("\n"), n)
}

/// Try to answer from memory hits. If we have a “needle”, return the best matching sentence.
/// Otherwise, return the top snippet. Returns (answer, trace_note).
fn answer_from_hits(query: &str, ctx: &ContextFrame) -> Option<(String, String)> {
    if ctx.top_hits.is_empty() {
        return None;
    }

    let needle_opt = extract_needle(query);

    // Try to find a precise answer near the needle first.
    if let Some(needle) = needle_opt {
        let needle_lc = needle.to_lowercase();

        for hit in ctx.top_hits.iter().take(5) {
            if let Some(sent) = best_sentence_for(&hit.text, &needle_lc) {
                // Try to extract the value after “is / = / :” near the needle
                if let Some(val) = extract_value_near_needle(&sent, &needle_lc) {
                    let ans = val.trim().to_string();
                    let trace = format!(
                        "Matched value for \"{}\" in: \"{}\" (sim {:.2}).",
                        needle, sent, hit.sim
                    );
                    return Some((ans, trace));
                }

                // If we couldn’t extract a tight value, return the whole sentence.
                let ans = sent.trim().to_string();
                let trace = format!(
                    "Matched sentence for needle \"{}\" (sim {:.2}).",
                    needle, hit.sim
                );
                return Some((ans, trace));
            }
        }
    }

    // Fallback: return a snippet of the best hit
    let top = &ctx.top_hits[0];
    let snippet = if top.text.len() > 220 {
        format!("{}…", &top.text[..220])
    } else {
        top.text.clone()
    };
    Some((snippet, format!("Returned top snippet (sim {:.2}).", top.sim)))
}

/// Extract a value appearing after common markers near the needle.
/// e.g. “… launch code is zebra-13” → “zebra-13”
fn extract_value_near_needle(sentence: &str, needle_lc: &str) -> Option<String> {
    let s = sentence.trim();
    let s_lc = s.to_lowercase();

    // Find the needle
    let npos = s_lc.find(needle_lc)?;
    // Search for a marker shortly after the needle
    let markers = [" is ", " are ", " = ", ": "];

    // Look in the tail starting at the needle
    let tail = &s[npos..];
    let tail_lc = &s_lc[npos..];

    // Find the first marker in the tail
    let mut start_idx = None;
    for m in markers.iter() {
        if let Some(mi) = tail_lc.find(m) {
            start_idx = Some(npos + mi + m.len());
            break;
        }
    }
    let start = start_idx?;

    // Slice the value until punctuation or line end
    let rest = &s[start..];
    let end_rel = rest.find(|c: char| c == '.' || c == '!' || c == '?').unwrap_or(rest.len());
    let mut val = rest[..end_rel].trim().to_string();

    // Light cleanup
    if let Some(stripped) = val.strip_prefix("called ").or_else(|| val.strip_prefix("named ")) {
        val = stripped.trim().to_string();
    }

    if val.is_empty() { None } else { Some(val) }
}


/* ------------ solver ------------ */

pub fn solve_with_context(
    mode: &CortexReasoningMode,
    frame: &CortexInputFrame,
    context: &ContextFrame,
) -> CortexDraftOutput {
    match mode {
        // Prefer internal recall: if we have hits, answer from memory; else fall back
        CortexReasoningMode::DirectRecall => {
            if !context.top_hits.is_empty() {
                if let Some((ans, note)) = answer_from_hits(frame.raw_input, context) {
                    let (_, n) = summarize_hits(context, 3);
                    return CortexDraftOutput {
                        response: Some(ans), // ← final answer
                        action: None,
                        trace: Some(format!("DirectRecall: used {n} internal hits. {note}")),
                    };
                }
                // no targeted sentence, but have hits → show short list
                let (summary, n) = summarize_hits(context, 3);
                CortexDraftOutput {
                    response: Some(summary),
                    action: None,
                    trace: Some(format!("DirectRecall: summarized {n} internal hits.")),
                }
            } else {
                // No useful internal context → gentle fallback
                let mut resp = String::from("No strong internal matches. ");
                resp.push_str("Would consider external search.");
                CortexDraftOutput {
                    response: Some(resp),
                    action: Some(CortexAction::SearchInternet(frame.raw_input.to_string())),
                    trace: Some("Fallback from DirectRecall → InternetResearch.".to_string()),
                }
            }
        }

        CortexReasoningMode::InternetResearch => CortexDraftOutput {
            response: Some("Conducting internet search...".to_string()),
            action: Some(CortexAction::SearchInternet(frame.raw_input.to_string())),
            trace: Some("InternetResearch selected by selector.".to_string()),
        },

        CortexReasoningMode::PerceptionSynthesis => CortexDraftOutput {
            response: Some("Analyzing environment...".to_string()),
            action: Some(CortexAction::ScanEnvironment),
            trace: Some("Perception synthesis routine.".to_string()),
        },

        CortexReasoningMode::AbstractSolve => {
            let resp = if !context.top_hits.is_empty() {
                let (summary, _) = summarize_hits(context, 2);
                format!("Abstract reasoning using retrieved context:\n{summary}")
            } else {
                "Solving problem abstractly...".to_string()
            };
            CortexDraftOutput {
                response: Some(resp),
                action: None,
                trace: Some("Applied abstract reasoning pattern.".to_string()),
            }
        }

        CortexReasoningMode::ChainOfThought => {
            // Build introspection bullets for trace
            let mut bullets = String::from("Thinking step by step…\n- First, recall the most relevant memory.\n");
            if let Some(top) = context.top_hits.first() {
                let snippet = if top.text.len() > 140 { format!("{}…", &top.text[..140]) } else { top.text.clone() };
                bullets.push_str(&format!("  → [{:.2}] {snippet}\n", top.sim));
            } else {
                bullets.push_str("  → No internal hits.\n");
            }
            bullets.push_str("- Next, relate it to the question and produce an answer.");

            // Try to actually answer from the hits
            if let Some((ans, explain)) = answer_from_hits(frame.raw_input, context) {
                CortexDraftOutput {
                    response: Some(ans), // ← final answer (what user sees)
                    action: None,
                    trace: Some(format!("{bullets}\n\n[explain] {explain}")),
                }
            } else {
                // No crisp answer; keep the helpful bullets as the response
                CortexDraftOutput {
                    response: Some(bullets),
                    action: None,
                    trace: Some("CoT: no extractable answer from hits.".to_string()),
                }
            }
        }

        CortexReasoningMode::ActionSynthesis => CortexDraftOutput {
            response: Some("Preparing an action plan...".to_string()),
            action: Some(CortexAction::RunSubmodule("plan_executor".into())),
            trace: Some("Synthesized action plan.".to_string()),
        },

        CortexReasoningMode::MetaReflect => CortexDraftOutput {
            response: Some("Reflecting on approach and next steps...".to_string()),
            action: None,
            trace: Some("Meta reflection drafted.".to_string()),
        },
    }
}

