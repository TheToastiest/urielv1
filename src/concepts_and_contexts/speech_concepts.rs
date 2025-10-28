/// Simple surface realizer for concept answers.
/// Keep this tiny and deterministic; we can learn/style-swap later.
pub fn realize_concept(key: &str, val: &str) -> String {
    match key {
        "name"        => format!("I’m {val}."),
        "you are"     => format!("I’m {val}."),
        "creator"     => format!("My creator is {val}."),
        "mentor"      => format!("My mentor is {val}."),
        "purpose"     => format!("My purpose is {val}."),
        "launch code" => val.to_string(), // codes should remain raw
        _             => val.to_string(),
    }
}
