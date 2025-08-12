use regex::Regex;

pub enum FactCmd<'a> {
    Store { key: &'a str, value: &'a str },
    Query { key: &'a str },
    None,
}

pub fn parse_fact(text: &str) -> FactCmd<'_> {
    let t = text.trim();

    // store this: X is Y
    // remember that X is Y
    let re_store = Regex::new(r#"(?i)^(store this:|remember that)\s*(.+?)\s+is\s+(.+)$"#).unwrap();
    if let Some(c) = re_store.captures(t) {
        let key = c.get(2).unwrap().as_str().trim();
        let val = c.get(3).unwrap().as_str().trim();
        return FactCmd::Store { key, value: val };
    }

    // what is X / what's X / who is X / where is X
    let re_what = Regex::new(r#"(?i)^(what(?:'s| is)|who is|where is)\s+(.+?)\??$"#).unwrap();
    if let Some(c) = re_what.captures(t) {
        let key = c.get(2).unwrap().as_str().trim();
        return FactCmd::Query { key };
    }

    FactCmd::None
}
