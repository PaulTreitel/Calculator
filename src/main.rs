mod tokenizer;
mod syntax_tree;

fn main() {
    println!("Hello, world!");
    let input = "3+(5*2.2^.4)";
    let tokens = tokenizer::tokenize(input);
    if tokens.is_err() {
        println!("Error on tokenization");
        return;
    }
    let tokens = tokens.ok().unwrap();
    if !tokenizer::is_valid_expr(&tokens) {
        println!("Tokens are not a valid expression");
        return;
    }
    let syntax_tree = syntax_tree::ExprTree::from_tokens(&tokens);
    if let Ok(val) = syntax_tree.evaluate() {
        println!("{}", val);
    } else {
        println!("Error on evaluation");
    }
}