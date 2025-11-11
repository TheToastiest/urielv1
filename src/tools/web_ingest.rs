use anyhow::Result;
use rusqlite::Connection;
use reqwest;
use html2text;

pub struct IngestCfg<'a> {
    pub whitelist_domains: &'a [&'a str],
    pub max_bytes_per_page: usize,   // e.g. 300_000
    pub max_chunks: usize,           // e.g. 100
}

pub fn ingest_urls(conn: &Connection, urls: &[&str], cfg: &IngestCfg) -> Result<()> {
    for url in urls {
        if !is_whitelisted(url, cfg.whitelist_domains) { continue; }

        // fetch
        let html = fetch_blocking(url, cfg.max_bytes_per_page)?;
        let text = html_to_text(&html);
        let chunks = chunk_text(&text, 800, cfg.max_chunks); // ~800 chars, hard cap

        // queue: turn chunks into learn_queue items (positive anchor → the chunk;
        // question can be the page title or first sentence; hardness mild)
        for (i, c) in chunks.into_iter().enumerate() {
            let _ = conn.execute(
                "INSERT INTO learn_queue(ts,question,positive,hardness,used)
               VALUES(strftime('%s','now'),?1,?2,?3,0)",
                rusqlite::params![format!("from {}", url), c, 0.35_f32 + (i as f32)*0.01],
            );
        }
    }
    Ok(())
}

// ── helpers (simple & safe) ────────────────────────────────────────────────

fn is_whitelisted(url: &str, whitelist: &[&str]) -> bool {
    let host = url::Url::parse(url).ok().and_then(|u| u.host_str().map(|s| s.to_string()));
    match host {
        Some(h) => whitelist.iter().any(|d| h.ends_with(d)),
        None    => false
    }
}

fn fetch_blocking(url: &str, max_bytes: usize) -> Result<String> {
    // add in Cargo.toml:
    // reqwest = { version = "0.11", features = ["blocking","rustls-tls"] }
    let resp = reqwest::blocking::get(url)?;
    let mut body = resp.text()?;
    if body.len() > max_bytes { body.truncate(max_bytes); }
    Ok(body)
}

fn html_to_text(html: &str) -> String {
    // html2text = "0.6" (light & pure)
    html2text::from_read(html.as_bytes(), 80)
}

fn chunk_text(s: &str, max_len: usize, cap: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || looks_like_boilerplate(line) { continue; }
        if buf.len() + 1 + line.len() > max_len {
            if !buf.is_empty() { out.push(std::mem::take(&mut buf)); }
            if out.len() >= cap { break; }
        }
        if !buf.is_empty() { buf.push(' '); }
        buf.push_str(line);
    }
    if !buf.is_empty() && out.len() < cap { out.push(buf); }
    out
}

fn looks_like_boilerplate(l: &str) -> bool {
    let ll = l.to_ascii_lowercase();
    // crude ad/privacy/footer filters
    ll.contains("cookie") || ll.contains("subscribe") || ll.contains("advert")
        || ll.starts_with("© ") || ll.starts_with("copyright")
        || ll.len() < 40 // super-short lines are often nav
}


// how to use// in main(), after opening `conn`
// let cfg = URIELV1::tools::web_ingest::IngestCfg {
//     whitelist_domains: &["example.com", "wikipedia.org"],
//     max_bytes_per_page: 300_000,
//     max_chunks: 80,
// };
// URIELV1::tools::web_ingest::ingest_urls(&conn, &["https://en.wikipedia.org/wiki/URIEL"], &cfg)?;