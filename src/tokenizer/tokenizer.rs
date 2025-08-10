// smie_core/src/tokenizer/vocab.rs

use std::collections::HashMap;

pub enum EncodingMode {
    Readable,
    Hex,
    Hybrid,
}

pub struct Vocab {
    pub char_to_index: HashMap<char, usize>,
    pub index_to_char: HashMap<usize, char>,
    pub encoding_mode: EncodingMode,
}

impl Vocab {
    pub fn new(mode: EncodingMode) -> Self {
        let charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}<>@#$%^&*-_=+|/\\"
            .chars()
            .collect::<Vec<_>>();

        let char_to_index = charset.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let index_to_char = charset.iter().enumerate().map(|(i, &c)| (i, c)).collect();

        Vocab {
            char_to_index,
            index_to_char,
            encoding_mode: mode,
        }
    }

    pub fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        match self.encoding_mode {
            EncodingMode::Readable => self.encode_readable(text, max_len),
            EncodingMode::Hex => self.encode_hex(text, max_len),
            EncodingMode::Hybrid => {
                let mut readable = self.encode_readable(text, max_len / 2);
                let mut hex = self.encode_hex(text, max_len / 2);
                readable.append(&mut hex);
                readable
            }
        }
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .filter_map(|i| self.index_to_char.get(i))
            .collect::<String>()
    }

    fn encode_readable(&self, text: &str, max_len: usize) -> Vec<usize> {
        let mut encoded = vec![0; max_len];
        for (i, c) in text.chars().take(max_len).enumerate() {
            encoded[i] = *self.char_to_index.get(&c).unwrap_or(&0);
        }
        encoded
    }

    fn encode_hex(&self, text: &str, max_len: usize) -> Vec<usize> {
        let hex_string = text.as_bytes().iter().map(|b| format!("{:02x}", b)).collect::<String>();
        let mut encoded = vec![0; max_len];
        for (i, c) in hex_string.chars().take(max_len).enumerate() {
            encoded[i] = *self.char_to_index.get(&c).unwrap_or(&0);
        }
        encoded
    }
}
