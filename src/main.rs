use syntax_tree::ExprTree;
use tokenizer::{is_valid_expr, tokenize};
use std::io::{self, Write};

mod tokenizer;
mod syntax_tree;

fn main() {
    let mut input= String::new();
    println!("Welcome to the calculator. Enter an expression or enter \"q\" to exit\n");
    loop {
        input.clear();
        print!(">>> ");
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut input)
            .expect("Failed to read line");
        input = input[0..input.len() - 1].to_string();
        if input.eq_ignore_ascii_case("q") {
            break;
        }
        println!();

        let tokens = tokenize(input.as_str());
        match &tokens {
            Ok(_) => (),
            Err(e) => {
                println!("Error: {}\n", e);
                continue;
            },
        }
        let tokens = tokens.ok().unwrap();
        match is_valid_expr(&tokens) {
            Ok(()) => (),
            Err(e) => {
                println!("Error: {}\n", e);
                continue;
            },
        }
        let syntax_tree = ExprTree::from_tokens(&tokens);
        match syntax_tree.evaluate() {
            Ok(val) => {
                println!("{}\n", val);
            },
            Err(e) => {
                println!("Error: {}\n", e);
            }
        }
    }
}