use clap::{Parser, Subcommand};
use color_eyre::Result;
use rusqlite::{ffi::sqlite3_auto_extension, Connection};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use sqlite_vss::{sqlite3_vector_init, sqlite3_vss_init};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Ingest { input: String },
    Summarize { query: String },
}

fn main() -> Result<()> {
    unsafe {
        sqlite3_auto_extension(Some(sqlite3_vector_init));
        sqlite3_auto_extension(Some(sqlite3_vss_init));
    }

    let db = Connection::open_in_memory()?;

    let opts = Args::parse();

    match &opts.command {
        Commands::Ingest { input } => {
            let model =
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
                    .create_model()?;

            // Define input
            let sentences = ["this is an example sentence", "each sentence is converted"];

            // Generate Embeddings
            let embeddings = model.encode(&sentences)?;
            println!("{embeddings:?}");
        }
        Commands::Summarize { query } => {
            todo!();
        }
    }

    let (version, vector): (String, String) = db.query_row(
        "SELECT vss_version(), vector_to_json(?)",
        [[0x00, 0x00, 0x28, 0x42]],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    println!("version={version} vector={vector}");

    db.execute_batch(
        r"
    CREATE VIRTUAL TABLE vss_demo USING vss0(a(2));
    INSERT INTO vss_demo(rowid, a)
      VALUES
          (1, '[1.0, 2.0]'),
          (2, '[2.0, 2.0]'),
          (3, '[3.0, 2.0]')
    ",
    )?;

    let result: Vec<(i64, f32)> = db
        .prepare(
            r"
          SELECT
            rowid,
            distance
          FROM vss_demo
          WHERE vss_search(a, '[1.0, 2.0]')
          LIMIT 3
        ",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;

    for (rowid, distance) in result {
        println!("rowid={rowid}, distance={distance}");
    }

    println!("✅ demo.rs ran successfully. \n");
    Ok(())
}
