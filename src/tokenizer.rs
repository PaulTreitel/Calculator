use logos::Logos;

const OPERATORS: [Token<'_>; 5] = [Token::Add, Token::Sub, Token::Mul, Token::Div, Token::Exp];

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")]
pub enum Token<'a> {
    #[regex("[+-]?(\\d*\\.)?\\d+")]
    Number(&'a str),

    #[regex("\\(|\\{|\\[")]
    OpenParen(&'a str),

    #[regex("\\)|\\]|\\}")]
    CloseParen(&'a str),

    #[token("+")]
    Add,

    #[token("-")]
    Sub,

    #[token("*")]
    Mul,

    #[token("/")]
    Div,

    #[token("^")]
    Exp,
}

impl<'a> Token<'a> {
    fn is_operator(&self) -> bool {
        OPERATORS.contains(self)
    }
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, ()> {
    let lex = Token::lexer(input);
    let mut all: Vec<Result<Token, ()>> = lex.collect();
    println!("{:?}", all);
    for idx in 0..all.len() {
        if all.get(idx).unwrap().clone().ok().is_none() {
            return Err(());
        }
    }
    let all: Vec<Token> = all.iter_mut()
        .map(|x| x.clone().ok().unwrap())
        .collect();
    Ok(all)
}

pub fn is_valid_expr(tokens: &Vec<Token>) -> bool {
    if has_unmatched_parens(tokens) {
        return false;
    }
    if has_empty_parens(tokens) {
        return false;
    }
    if has_start_end_ops(tokens) {
        return false;
    }
    if has_ops_without_operands(tokens) {
        return false;
    }
    // todo!("Validate the Input!");
    true
}

fn has_unmatched_parens(tokens: &Vec<Token>) -> bool {
    let mut open_parens = Vec::new();
    for token in tokens {
        match token {
            Token::OpenParen(open) => {
                open_parens.push(open);
            },
            Token::CloseParen(close) => {
                let open = open_parens.pop();
                if open.is_none() {
                    return true;
                }
                if open.unwrap().ne(close) {
                    return true;
                }
            },
            _ => (),
        }
    }
    false
}

fn has_empty_parens(tokens: &Vec<Token>) -> bool {
    let mut at_start = false;
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        match token {
            Token::OpenParen(_) => {
                at_start = true;
            },
            Token::CloseParen(_) => {
                if at_start {
                    return true;
                }
                at_start = false;
            }
            _ => {
                at_start = false;
            },
        }
    }
    false
}

fn has_start_end_ops(tokens: &Vec<Token>) -> bool {
    let mut at_start = true;
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        if at_start && token.is_operator() {
            return true;
        }
        match token {
            Token::OpenParen(_) => {
                at_start = true;
            },
            _ => {
                at_start = false;
            },
        }
    }
    let token = tokens.get(tokens.len() - 1).unwrap();
    token.is_operator()
}

fn has_ops_without_operands(tokens: &Vec<Token>) -> bool {
    for idx in 0..tokens.len() {
        let token = tokens.get(idx).unwrap();
        if !token.is_operator() {
            continue;
        }
        // guaranteed these will exist because this is validated after has_start_end_ops
        let prev = tokens.get(idx - 1).unwrap();
        let next = tokens.get(idx + 1).unwrap();
        match prev {
            Token::CloseParen(_) => (),
            Token::Number(_) => (),
            _ => {
                return true;
            },
        }
        match next {
            Token::OpenParen(_) => (),
            Token::Number(_) => (),
            _ => {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tokenization() {
        let input = "3+(5*2.2^.4]}{";
        let tokens = tokenize(input).ok().unwrap();
        assert_eq!(tokens, vec![
            Token::Number("3"),
            Token::Add,
            Token::OpenParen("("),
            Token::Number("5"),
            Token::Mul,
            Token::Number("2.2"),
            Token::Exp,
            Token::Number(".4"),
            Token::CloseParen("]"),
            Token::CloseParen("}"),
            Token::OpenParen("{"),
        ]);
    }

    #[test]
    fn open_period() {
        let input = "3+.";
        let tokens = tokenize(input);
        assert_eq!(tokens, Err(()));
    }

    #[test]
    fn letters() {
        let input = "3a+2";
        let tokens = tokenize(input);
        assert_eq!(tokens, Err(()));
    }
}