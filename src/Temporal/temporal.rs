use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct ParsedCommand {
    pub addressed_to: bool,
    pub wake_word: Option<String>,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub timestamp: u64,
    pub drive_alignment: HashMap<String, f32>, // Optional semantic weights
}

/// Returns the current UNIX timestamp in seconds.
fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Parses an input phrase and detects wake phrase, command, arguments, and drive relevance.
pub fn parse_phrase(input: &str, known_drives: &[&str]) -> ParsedCommand {
    let input_lower = input.to_lowercase();
    let tokens: Vec<&str> = input_lower.split_whitespace().collect();
    let timestamp = current_unix_timestamp();

    // Wake phrase matching
    let wake_keywords = ["uriel", "hey uriel", "listen uriel"];
    let addressed = wake_keywords.iter().any(|&w| input_lower.starts_with(w));

    // Determine command section after wake word
    let command_tokens = if addressed && tokens.len() > 1 {
        if tokens[0] == "uriel" && tokens[1] == "," {
            tokens[2..].to_vec()
        } else {
            tokens[1..].to_vec()
        }
    } else {
        tokens.clone()
    };

    // Command is first token after wake
    let command = command_tokens.first().map(|s| s.to_string());
    let args: Vec<String> = command_tokens.iter().skip(1).map(|s| s.to_string()).collect();

    // Drive alignment scoring
    let mut drive_alignment = HashMap::new();
    for &drive in known_drives {
        let mut score = 0.0;
        if input_lower.contains(drive) {
            score += 0.6;
        }
        if command.as_ref().map_or(false, |c| c.contains(drive)) {
            score += 0.4;
        }
        if score > 0.0 {
            drive_alignment.insert(drive.to_string(), score);
        }
    }

    ParsedCommand {
        addressed_to: addressed,
        wake_word: if addressed { Some("URIEL".to_string()) } else { None },
        command,
        args,
        timestamp,
        drive_alignment,
    }
}
