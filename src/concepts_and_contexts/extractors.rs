use super::concept::{Concept, ConceptHit};
use regex::Regex;

pub struct NameConcept;
impl Concept for NameConcept {
    fn key(&self) -> &'static str { "name" }
    fn extract_from_statement(&self, s: &str) -> Vec<ConceptHit> {
        let re = Regex::new(r"(?i)^\s*your\s+name\s+is\s+(.+?)\s*[.!]?\s*$").unwrap();
        re.captures(s).map(|cap| vec![ConceptHit{
            key: "name", value: cap[1].trim().into(), confidence: 0.95 }]).unwrap_or_default()
    }
    fn normalize_question(&self, q: &str) -> Option<String> {
        let l = q.trim().to_ascii_lowercase();
        if l.starts_with("what is your name") { Some("your name is".into()) } else { None }
    }
}

fn normalize_code(s:&str)->String{
    let digits:String = s.chars().filter(|c|c.is_ascii_digit()).collect();
    if digits.len()==9 { format!("{}-{}-{}", &digits[0..3],&digits[3..6],&digits[6..9]) } else { s.trim().into() }
}
pub struct LaunchCodeConcept;
impl Concept for LaunchCodeConcept {
    fn key(&self) -> &'static str { "launch code" }
    fn extract_from_statement(&self, s:&str)->Vec<ConceptHit>{
        let re = Regex::new(r"(?i)^\s*the\s+launch\s+code\s+is\s+([0-9][0-9\-\s]{7,})").unwrap();
        re.captures(s).map(|c| vec![ConceptHit{
            key:"launch code", value: normalize_code(&c[1]), confidence:0.95 }]).unwrap_or_default()
    }
    fn normalize_question(&self,q:&str)->Option<String>{
        let l=q.trim().to_ascii_lowercase();
        if l.starts_with("what is the launch code"){ Some("the launch code is".into()) } else { None }
    }
}

pub struct CreatorConcept;
impl Concept for CreatorConcept {
    fn key(&self)->&'static str { "creator" }
    fn extract_from_statement(&self,s:&str)->Vec<ConceptHit>{
        let re=Regex::new(r"(?i)^\s*(?:your|my)\s+creator\s+is\s+(.+?)\s*[.!]?\s*$").unwrap();
        if let Some(c)=re.captures(s){ return vec![ConceptHit{ key:"creator", value:c[1].trim().into(), confidence:0.9 }]; }
        let re2=Regex::new(r"(?i)^\s*i\s+am\s+your\s+([a-z\s]+?),\s*([A-Za-z][A-Za-z .'\-]+)").unwrap();
        if let Some(c)=re2.captures(s){
            if c[1].to_ascii_lowercase().contains("creator"){
                return vec![ConceptHit{ key:"creator", value:c[2].trim().into(), confidence:0.85 }];
            }
        }
        vec![]
    }
    fn normalize_question(&self,q:&str)->Option<String>{
        let l=q.trim().to_ascii_lowercase();
        if l.starts_with("who is your creator"){ Some("your creator is".into()) } else { None }
    }
}

pub struct MentorConcept;
impl Concept for MentorConcept {
    fn key(&self)->&'static str { "mentor" }
    fn extract_from_statement(&self,s:&str)->Vec<ConceptHit>{
        let re=Regex::new(r"(?i)^\s*(?:your|my)\s+mentor\s+is\s+(.+?)\s*[.!]?\s*$").unwrap();
        if let Some(c)=re.captures(s){ return vec![ConceptHit{ key:"mentor", value:c[1].trim().into(), confidence:0.9 }]; }
        let re2=Regex::new(r"(?i)^\s*i\s+am\s+your\s+([a-z\s]+?),\s*([A-Za-z][A-Za-z .'\-]+)").unwrap();
        if let Some(c)=re2.captures(s){
            if c[1].to_ascii_lowercase().contains("mentor"){
                return vec![ConceptHit{ key:"mentor", value:c[2].trim().into(), confidence:0.85 }];
            }
        }
        vec![]
    }
    fn normalize_question(&self,q:&str)->Option<String>{
        let l=q.trim().to_ascii_lowercase();
        if l.starts_with("who is your mentor"){ Some("your mentor is".into()) } else { None }
    }
}
