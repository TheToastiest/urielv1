// src/bin/uriel_sql.rs
use anyhow::{anyhow, Result};
use clap::Parser;
use rusqlite::{types::ValueRef, Connection, Row};
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(name="uriel_sql", version, about="Run an SQL query against the URIEL DB and print results")]
struct Args {
    /// SQL to execute, e.g. "SELECT * FROM learn_queue LIMIT 5;"
    #[arg(long)]
    sql: String,

    /// Output format: tsv or csv
    #[arg(long, default_value = "tsv")]
    format: String,

    /// DB path
    #[arg(long, default_value = "data/semantic_memory.db")]
    db: String,
}

fn display_cell(row: &Row, i: usize) -> String {
    match row.get_ref(i) {
        Ok(ValueRef::Null) => "".into(),
        Ok(ValueRef::Integer(n)) => n.to_string(),
        Ok(ValueRef::Real(x)) => x.to_string(),
        Ok(ValueRef::Text(bytes)) => String::from_utf8_lossy(bytes).to_string(),
        Ok(ValueRef::Blob(b)) => format!("<blob {} bytes>", b.len()),
        Err(e) => format!("<err {e}>"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut sep = '\t';
    let fmt = args.format.to_ascii_lowercase();
    if fmt == "csv" {
        sep = ',';
    }

    let conn = Connection::open(&args.db)?;
    let mut stmt = conn.prepare(&args.sql).map_err(|e| anyhow!("prepare failed: {e}"))?;

    // capture metadata BEFORE starting rows() to avoid borrow conflicts
    let col_count = stmt.column_count();
    let col_names: Vec<String> = (0..col_count)
        .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
        .collect();

    let mut rows = stmt.query([])?;

    // header
    {
        let mut line = String::new();
        for (i, name) in col_names.iter().enumerate() {
            if i > 0 {
                line.push(sep);
            }
            if sep == ',' {
                // minimal CSV escaping
                let needs_quote = name.contains(',') || name.contains('"') || name.contains('\n') || name.contains('\t');
                if needs_quote {
                    line.push('"');
                    line.push_str(&name.replace('"', "\"\""));
                    line.push('"');
                } else {
                    line.push_str(name);
                }
            } else {
                line.push_str(name);
            }
        }
        line.push('\n');
        io::stdout().write_all(line.as_bytes())?;
    }

    // rows
    while let Some(row) = rows.next()? {
        let mut line = String::new();
        for i in 0..col_count {
            if i > 0 {
                line.push(sep);
            }
            let cell = display_cell(row, i);
            if sep == ',' {
                let needs_quote = cell.contains(',') || cell.contains('"') || cell.contains('\n') || cell.contains('\t');
                if needs_quote {
                    line.push('"');
                    line.push_str(&cell.replace('"', "\"\""));
                    line.push('"');
                } else {
                    line.push_str(&cell);
                }
            } else {
                line.push_str(&cell);
            }
        }
        line.push('\n');
        io::stdout().write_all(line.as_bytes())?;
    }

    Ok(())
}
