// smie_core/src/tokenizer/vocab.rs

use std::collections::HashMap;

pub struct Vocab {
    pub char_to_index: HashMap<char, usize>,
    pub index_to_char: HashMap<usize, char>,
}

impl Vocab {
    pub fn new() -> Self {
        let charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}<>@#$%^&*-_=+|/\\"
            .chars()
            .collect::<Vec<_>>();

        let char_to_index = charset.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let index_to_char = charset.iter().enumerate().map(|(i, &c)| (i, c)).collect();

        Vocab {
            char_to_index,
            index_to_char,
        }
    }

    pub fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        let mut encoded = vec![0; max_len];
        for (i, c) in text.chars().take(max_len).enumerate() {
            encoded[i] = *self.char_to_index.get(&c).unwrap_or(&0);
        }
        encoded
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .filter_map(|i| self.index_to_char.get(i))
            .collect::<String>()
    }
}
